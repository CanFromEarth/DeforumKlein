"""
generate.py - FLUX.2-klein-4B image generation for Deforum

Uses diffusers Flux2KleinPipeline. Every frame is txt2img, then
blended with the warped previous frame for animation coherence.
Klein is distilled and doesn't support classical latent-space img2img.
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


def generate(args, root, frame=0, return_latent=False, return_sample=False, return_c=False):
    """Generate an image using FLUX.2-klein-4B."""
    seed_everything(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    pipe = root.pipe
    device = torch.device(root.device) if isinstance(root.device, str) else root.device
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    cond_prompt = args.cond_prompt
    assert cond_prompt is not None

    results = []

    with torch.no_grad():
        output = pipe(
            prompt=cond_prompt,
            height=args.H,
            width=args.W,
            num_inference_steps=args.steps,
            guidance_scale=args.scale,
            generator=generator,
            output_type="pt",
        )
        x_samples = output.images * 2.0 - 1.0  # [0,1] → [-1,1]
        x_samples = x_samples.to(device)

        # Blend with warped previous frame for animation coherence
        # strength controls how much new content vs previous frame:
        #   strength=1.0 → fully new (first frame)
        #   strength=0.3 → 30% new + 70% warped previous
        if args.init_sample is not None and args.strength < 1.0:
            x_samples = args.strength * x_samples + (1.0 - args.strength) * args.init_sample.to(device)

        # --- Collect results ---

        if return_latent:
            results.append(None)

        if args.use_mask and args.overlay_mask:
            init_image_tensor = None
            if args.init_sample_raw is not None:
                init_image_tensor = args.init_sample_raw
            elif args.init_sample is not None:
                init_image_tensor = args.init_sample

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
