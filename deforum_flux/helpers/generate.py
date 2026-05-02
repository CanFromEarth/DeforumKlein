"""
generate.py - FLUX.2-klein-4B image generation for Deforum

Uses diffusers Flux2KleinPipeline. Provides generate() / add_noise()
interface for render.py.

img2img via flow-matching noise injection.
"""

import os
import torch
import numpy as np
from PIL import Image
from pytorch_lightning import seed_everything
from einops import rearrange, repeat
from .load_images import load_img, prepare_overlay_mask


def add_noise(sample: torch.Tensor, noise_amt: float) -> torch.Tensor:
    return sample + torch.randn(sample.shape, device=sample.device) * noise_amt


def uint_number(datum, number):
    if number == 8:
        datum = Image.fromarray(datum.astype(np.uint8))
    elif number == 32:
        datum = datum.astype(np.float32)
    else:
        datum = datum.astype(np.uint16)
    return datum


def _vae_encode(pipe, image_tensor):
    """Encode image tensor [B, C, H, W] in [-1, 1] to scaled latent space."""
    vae = pipe.vae
    x = image_tensor.to(device=vae.device, dtype=vae.dtype)
    encoded = vae.encode(x)

    if hasattr(encoded, 'latent_dist'):
        latent = encoded.latent_dist.sample()
    elif hasattr(encoded, 'latents'):
        latent = encoded.latents
    else:
        latent = encoded[0]

    cfg = vae.config
    if getattr(cfg, 'scaling_factor', None):
        latent = latent * cfg.scaling_factor
    if getattr(cfg, 'shift_factor', None):
        latent = latent - cfg.shift_factor

    return latent


def _fold_patches(latents):
    """Fold 2x2 spatial patches into channels: (B,C,H,W) → (B,C*4,H/2,W/2).

    This is the format Klein's prepare_latents expects for custom latents.
    """
    B, C, H, W = latents.shape
    latents = latents.view(B, C, H // 2, 2, W // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)  # (B, C, 2, 2, H/2, W/2)
    latents = latents.reshape(B, C * 4, H // 2, W // 2)
    return latents


def _vae_decode(pipe, latent):
    """Decode scaled latent to image tensor [B, C, H, W]."""
    vae = pipe.vae
    cfg = vae.config

    x = latent.to(device=vae.device, dtype=vae.dtype)

    if getattr(cfg, 'shift_factor', None):
        x = x + cfg.shift_factor
    if getattr(cfg, 'scaling_factor', None):
        x = x / cfg.scaling_factor

    decoded = vae.decode(x)
    return decoded.sample if hasattr(decoded, 'sample') else decoded[0]


def _tensor_to_pil(tensor):
    """Convert [1, C, H, W] tensor in [-1, 1] to PIL Image."""
    img = tensor[0].clamp(-1, 1).permute(1, 2, 0).cpu().float().numpy()
    img = ((img * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)


def generate(args, root, frame=0, return_latent=False, return_sample=False, return_c=False):
    """Generate an image using FLUX.2-klein-4B."""
    seed_everything(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    pipe = root.pipe
    device = torch.device(root.device) if isinstance(root.device, str) else root.device
    dtype = torch.bfloat16
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    batch_size = args.n_samples
    if batch_size > 1:
        raise NotImplementedError("Batch size > 1 not supported yet")

    # Pipeline expects latents as (B, C*4, H/2, W/2) — patches folded into channels.
    # For 1024x1024, vae_scale=8: latent spatial = 128x128, folded = (128, 64, 64)
    num_ch = pipe.transformer.config.in_channels // 4  # 32
    vae_scale = pipe.vae_scale_factor  # 8
    h_lat = 2 * (args.H // (vae_scale * 2))  # 128
    w_lat = 2 * (args.W // (vae_scale * 2))   # 128
    folded_shape = (batch_size, num_ch * 4, h_lat // 2, w_lat // 2)  # (1, 128, 64, 64)

    # --- Resolve init image / latent ---
    init_latent = None
    init_image_tensor = None

    if args.init_latent is not None:
        init_latent = args.init_latent.to(device, dtype=dtype)

    elif args.init_sample is not None:
        with torch.no_grad():
            init_latent = _vae_encode(pipe, args.init_sample.to(torch.float32))
            init_latent = init_latent.to(device=device, dtype=dtype)

    elif args.use_init and args.init_image is not None and args.init_image != '':
        loaded_img, mask_image = load_img(
            args.init_image,
            shape=(args.W, args.H),
            use_alpha_as_mask=args.use_alpha_as_mask
        )
        if args.add_init_noise:
            loaded_img = add_noise(loaded_img, args.init_noise)
        init_image_tensor = loaded_img.to(device)
        with torch.no_grad():
            init_latent = _vae_encode(pipe, init_image_tensor.to(torch.float32))
            init_latent = init_latent.to(device=device, dtype=dtype)

    if not args.use_init and args.strength > 0 and args.strength_0_no_init:
        args.strength = 0

    cond_prompt = args.cond_prompt
    assert cond_prompt is not None

    results = []

    with torch.no_grad():
        if init_latent is not None and args.strength > 0:
            # === IMG2IMG — flow-matching noise injection ===

            # Fold VAE latent (B,C,H,W) → (B,C*4,H/2,W/2) to match pipeline format
            init_latent_folded = _fold_patches(init_latent)

            # Generate noise in the folded shape
            noise = torch.randn(
                folded_shape, dtype=dtype,
                generator=torch.Generator(device="cpu").manual_seed(args.seed),
            ).to(device=device)

            t = args.strength
            noisy_latent = t * noise + (1.0 - t) * init_latent_folded

            output = pipe(
                prompt=cond_prompt,
                height=args.H,
                width=args.W,
                num_inference_steps=args.steps,
                guidance_scale=args.scale,
                latents=noisy_latent,
                generator=generator,
                output_type="pt",
            )

            x_samples = output.images * 2.0 - 1.0  # [0,1] → [-1,1]

        else:
            # === TXT2IMG — first frame or strength=0 ===
            output = pipe(
                prompt=cond_prompt,
                height=args.H,
                width=args.W,
                num_inference_steps=args.steps,
                guidance_scale=args.scale,
                generator=generator,
                output_type="pt",
            )
            x_samples = output.images * 2.0 - 1.0

        x_samples = x_samples.to(device)

        # --- Collect results ---

        if return_latent:
            lat = _vae_encode(pipe, x_samples.to(torch.float32))
            results.append(lat)

        if args.use_mask and args.overlay_mask:
            if args.init_sample_raw is not None:
                img_original = args.init_sample_raw
            elif init_image_tensor is not None:
                img_original = init_image_tensor
            else:
                raise Exception("Cannot overlay mask without an init image")

            if args.mask_sample is None or getattr(args, 'using_vid_init', False):
                args.mask_sample = prepare_overlay_mask(args, root, img_original.shape)

            x_samples = img_original * args.mask_sample + x_samples * ((args.mask_sample * -1.0) + 1)

        if return_sample:
            results.append(x_samples.clone())

        if return_c:
            print("[generate] WARNING: return_c (interpolation) not yet supported")
            results.append(None)

        x_samples_np = x_samples.clamp(-1, 1)
        x_samples_np = rearrange(x_samples_np[0], "c h w -> h w c")
        image = (127.5 * (x_samples_np + 1.0)).cpu().byte().numpy()
        image = uint_number(image, args.bit_depth_output)
        results.append(image)

    return results
