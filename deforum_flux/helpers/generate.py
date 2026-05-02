"""
generate.py - FLUX.2-klein-4B image generation for Deforum

Uses Flux2KleinPipeline with native image conditioning.
Frame 0: txt2img. Frame 1+: warped previous frame as reference image.
"""

import os
import torch
import numpy as np
from PIL import Image
from pytorch_lightning import seed_everything
from einops import rearrange
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


def _tensor_to_pil(tensor):
    """Convert [1, C, H, W] tensor in [-1, 1] to PIL Image."""
    img = tensor[0].clamp(-1, 1).permute(1, 2, 0).cpu().float().numpy()
    img = ((img * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)


def generate(args, root, frame=0, return_latent=False, return_sample=False, return_c=False):
    """Generate an image using FLUX.2-klein-4B with native image conditioning."""
    seed_everything(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    pipe = root.pipe
    device = torch.device(root.device) if isinstance(root.device, str) else root.device
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    cond_prompt = args.cond_prompt
    assert cond_prompt is not None

    # Build reference image from warped previous frame
    ref_image = None
    if args.init_sample is not None:
        ref_image = _tensor_to_pil(args.init_sample)

    results = []

    with torch.no_grad():
        pipe_kwargs = dict(
            prompt=cond_prompt,
            height=args.H,
            width=args.W,
            num_inference_steps=args.steps,
            guidance_scale=args.scale,
            generator=generator,
            output_type="pt",
        )

        if ref_image is not None:
            pipe_kwargs["image"] = ref_image

        output = pipe(**pipe_kwargs)
        x_samples = output.images * 2.0 - 1.0  # [0,1] → [-1,1]
        x_samples = x_samples.to(device)

        # --- Collect results ---

        if return_latent:
            results.append(None)

        if args.use_mask and args.overlay_mask:
            init_image_tensor = args.init_sample_raw or args.init_sample
            if init_image_tensor is not None:
                if args.mask_sample is None or getattr(args, 'using_vid_init', False):
                    args.mask_sample = prepare_overlay_mask(args, root, init_image_tensor.shape)
                x_samples = init_image_tensor * args.mask_sample + x_samples * ((args.mask_sample * -1.0) + 1)

        if return_sample:
            results.append(x_samples.clone())

        if return_c:
            results.append(None)

        x_samples_np = x_samples.clamp(-1, 1)
        x_samples_np = rearrange(x_samples_np[0], "c h w -> h w c")
        image = (127.5 * (x_samples_np + 1.0)).cpu().byte().numpy()
        image = uint_number(image, args.bit_depth_output)
        results.append(image)

    return results
