"""
generate.py - FLUX.2-klein-4B image generation for Deforum

Uses diffusers Flux2KleinPipeline. Provides generate() / add_noise()
interface for render.py.

img2img via flow-matching noise injection:
  z_t = strength * noise + (1 - strength) * encode(prev_frame)
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


def _pack_latents(latents):
    """Pack (B, C, H, W) → (B, H/2*W/2, C*4) for FLUX transformer."""
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(b, (h // 2) * (w // 2), c * 4)
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
            noise = torch.randn(
                init_latent.shape, dtype=dtype,
                generator=torch.Generator(device="cpu").manual_seed(args.seed),
            ).to(device=device)

            t = args.strength
            noisy_latent = t * noise + (1.0 - t) * init_latent

            try:
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
            except TypeError:
                # Fallback: pass previous frame as reference image
                print("[generate] 'latents' not supported, using reference image fallback")
                ref_pil = _tensor_to_pil(
                    _vae_decode(pipe, init_latent) if init_image_tensor is None
                    else init_image_tensor
                )
                output = pipe(
                    prompt=cond_prompt,
                    image=[ref_pil],
                    height=args.H,
                    width=args.W,
                    num_inference_steps=args.steps,
                    guidance_scale=args.scale,
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
