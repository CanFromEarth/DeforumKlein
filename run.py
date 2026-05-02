#!/usr/bin/env python3
"""
Deforum animation with FLUX.2-klein-4B

Usage:
    python run.py

    # If model is gated:
    export HF_TOKEN=hf_xxxxxxxxxxxxx
    python run.py
"""

# %%
# --------------------------------------------------------------------------- #
#  Auto-install missing dependencies
# --------------------------------------------------------------------------- #
import subprocess, sys, os

def _ensure_installed(package, pip_name=None):
    try:
        __import__(package)
    except ImportError:
        pip_name = pip_name or package
        print(f"Installing {pip_name} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])

def _ensure_diffusers():
    try:
        from diffusers import Flux2KleinPipeline  # noqa: F401
        return
    except ImportError:
        pass
    print("Installing diffusers (latest, with FLUX.2 support) ...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "git+https://github.com/huggingface/diffusers.git"
    ])

_ensure_installed("accelerate")
_ensure_installed("scipy")
_ensure_diffusers()

if os.system("which ffmpeg > /dev/null 2>&1") != 0:
    print("Installing ffmpeg ...")
    os.system("apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null 2>&1")

# --------------------------------------------------------------------------- #
#  Patch IPython display for headless (RunPod / SSH terminal)
# --------------------------------------------------------------------------- #
try:
    from IPython import get_ipython
    if get_ipython() is None:
        raise RuntimeError("not in IPython")
except Exception:
    import types
    _display_mod = types.ModuleType("IPython.display")
    _display_mod.display = lambda *a, **kw: None
    _display_mod.clear_output = lambda *a, **kw: None
    sys.modules["IPython.display"] = _display_mod
    if "IPython" not in sys.modules:
        _ip_mod = types.ModuleType("IPython")
        sys.modules["IPython"] = _ip_mod
    sys.modules["IPython"].display = _display_mod

# %%
# --------------------------------------------------------------------------- #
#  Imports
# --------------------------------------------------------------------------- #
import time, gc, random
import torch
from types import SimpleNamespace

sys.path.extend(['./deforum_flux', './deforum_flux/src'])

from helpers.save_images import get_output_folder
from helpers.settings import load_args
from helpers.render import (
    render_animation, render_input_video,
    render_image_batch, render_interpolation
)
from helpers.prompts import Prompts
from helpers.ffmpeg_helpers import (
    get_extension_maxframes,
    get_auto_outdir_timestring,
    get_ffmpeg_path,
    make_mp4_ffmpeg,
)


# %%
# --------------------------------------------------------------------------- #
#  FFmpeg helper
# --------------------------------------------------------------------------- #
def ffmpegArgs():
    ffmpeg_mode = "auto"
    ffmpeg_outdir = ""
    ffmpeg_timestring = ""
    ffmpeg_image_path = ""
    ffmpeg_mp4_path = ""
    ffmpeg_gif_path = ""
    ffmpeg_extension = "png"
    ffmpeg_maxframes = 200
    ffmpeg_fps = 12

    if ffmpeg_mode == 'auto':
        ffmpeg_outdir, ffmpeg_timestring = get_auto_outdir_timestring(args, ffmpeg_mode)
    if ffmpeg_mode in ["auto", "timestring"]:
        ffmpeg_extension, ffmpeg_maxframes = get_extension_maxframes(args, ffmpeg_outdir, ffmpeg_timestring)
        ffmpeg_image_path, ffmpeg_mp4_path, ffmpeg_gif_path = get_ffmpeg_path(
            ffmpeg_outdir, ffmpeg_timestring, ffmpeg_extension
        )
    return locals()


# %%
# --------------------------------------------------------------------------- #
#  Model — FLUX.2-klein-4B via diffusers
# --------------------------------------------------------------------------- #
class Model:
    def __init__(self, model_id="black-forest-labs/FLUX.2-klein-4B", offload=True):
        from diffusers import Flux2KleinPipeline

        hf_token = os.environ.get("HF_TOKEN", None)
        if hf_token:
            print("Using HF_TOKEN for authentication")

        print(f"Loading {model_id} ...")
        self.pipe = Flux2KleinPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            token=hf_token,
        )

        if offload:
            self.pipe.enable_model_cpu_offload()
            print("CPU offload enabled (~13 GB VRAM)")
        else:
            self.pipe = self.pipe.to("cuda")
            print("Full GPU mode")


# %%
# --------------------------------------------------------------------------- #
#  Path & Model setup
# --------------------------------------------------------------------------- #
def PathSetup():
    output_path = "outputs"
    models_path = "./models"
    return locals()

root = SimpleNamespace(**PathSetup())


def ModelSetup():
    map_location = "cuda"
    device = torch.device(map_location)

    gpu_mem = 0
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    use_offload = gpu_mem < 30

    model = Model(offload=use_offload)
    pipe = model.pipe
    return locals()

root.__dict__.update(ModelSetup())


# %%
# --------------------------------------------------------------------------- #
#  Animation settings
# --------------------------------------------------------------------------- #
def DeforumAnimArgs():
    animation_mode = '2D'
    max_frames = 120
    border = 'replicate'

    angle = "0:(0)"
    zoom = "0:(1.04)"
    translation_x = "0:(0)"
    translation_y = "0:(0)"
    translation_z = "0:(7.5)"
    rotation_3d_x = "0:(0)"
    rotation_3d_y = "0:(0)"
    rotation_3d_z = "0:(0)"
    flip_2d_perspective = False
    perspective_flip_theta = "0:(0)"
    perspective_flip_phi = "0:(t%15)"
    perspective_flip_gamma = "0:(0)"
    perspective_flip_fv = "0:(53)"
    noise_schedule = "0: (0.02)"

    # Strength schedule — tuned for 8 steps (granularity ~ 0.125)
    strength_schedule = "0: (0.625), 12: (0.625), 24: (0.75), 36: (0.75), 38: (0.625)"

    contrast_schedule = "0: (1.0)"
    hybrid_comp_alpha_schedule = "0:(1)"
    hybrid_comp_mask_blend_alpha_schedule = "0:(0.5)"
    hybrid_comp_mask_contrast_schedule = "0:(1)"
    hybrid_comp_mask_auto_contrast_cutoff_high_schedule = "0:(100)"
    hybrid_comp_mask_auto_contrast_cutoff_low_schedule = "0:(0)"

    enable_schedule_samplers = False
    sampler_schedule = "0:('Default Scheduler')"

    kernel_schedule = "0: (5)"
    sigma_schedule = "0: (1.0)"
    amount_schedule = "0: (0.2)"
    threshold_schedule = "0: (0.0)"

    color_coherence = 'Match Frame 0 RGB'
    color_coherence_video_every_N_frames = 1
    color_force_grayscale = False
    diffusion_cadence = '1'

    use_depth_warping = True
    midas_weight = 0.3
    near_plane = 200
    far_plane = 10000
    fov = 40
    padding_mode = 'border'
    sampling_mode = 'bicubic'
    save_depth_maps = False

    video_init_path = '/content/video_in.mp4'
    extract_nth_frame = 1
    overwrite_extracted_frames = True
    use_mask_video = False
    video_mask_path = '/content/video_in.mp4'

    hybrid_generate_inputframes = False
    hybrid_use_first_frame_as_init_image = True
    hybrid_motion = "None"
    hybrid_motion_use_prev_img = False
    hybrid_flow_method = "DIS Medium"
    hybrid_composite = False
    hybrid_comp_mask_type = "None"
    hybrid_comp_mask_inverse = False
    hybrid_comp_mask_equalize = "None"
    hybrid_comp_mask_auto_contrast = False
    hybrid_comp_save_extra_frames = False
    hybrid_use_video_as_mse_image = False

    interpolate_key_frames = False
    interpolate_x_frames = 32

    resume_from_timestring = False
    resume_timestring = "20240810001544"

    return locals()


# %%
# --------------------------------------------------------------------------- #
#  Prompts
# --------------------------------------------------------------------------- #
prompts = {
    0: "super realism, 4k, a highly detailed close-up view of a woman's mesmerizing blue eye, with realistic reflections and an intense natural sparkle. The iris displays intricate patterns of deep blues and subtle hints of lighter hues, while delicate veins add to the eye's natural complexity. Soft, diffused lighting enhances the eye's depth, with a blurred background to emphasize the eye's captivating beauty and detail.",
    12: "super realism, 4k, the woman's blue eye transforms into a stunning cosmic scene. Tiny, luminous stars begin to appear within the iris, creating a sense of depth. Nebulae with swirling, ethereal colors\u2014rich purples, blues, and pinks\u2014emerge, blending seamlessly with the natural textures of the eye.",
    24: "super realism, 4k, grand cosmic vista contained within the eye. The eye now features swirling galaxies with vibrant, spiraling arms, and floating celestial bodies such as distant planets and shimmering asteroids.",
    36: "super realism, 4k, The swirling galaxies and celestial bodies are now accompanied by pulsating stars and radiant supernovae, with intricate light effects and a sense of motion.",
}

# Distilled model: no CFG, negative prompts are ignored.
neg_prompts = {
    0: "",
}


# %%
# --------------------------------------------------------------------------- #
#  Generation settings
# --------------------------------------------------------------------------- #
def DeforumArgs():
    W = 1024
    H = 1024
    W, H = map(lambda x: x - x % 64, (W, H))
    bit_depth_output = 8

    seed = -1
    sampler = 'Default Scheduler'
    steps = 8
    scale = 1.0
    dynamic_threshold = None
    static_threshold = None

    save_samples = True
    save_settings = True
    display_samples = False
    save_sample_per_step = False
    show_sample_per_step = False

    n_batch = 1
    n_samples = 1
    batch_name = "DeforumKlein"
    filename_format = "{timestring}_{index}_{prompt}.png"
    seed_behavior = "iter"
    seed_iter_N = 1
    make_grid = False
    grid_rows = 2
    outdir = get_output_folder(root.output_path, batch_name)

    use_init = False
    strength = 1.0
    strength_0_no_init = True
    init_image = ""
    add_init_noise = False
    init_noise = 0.01
    use_mask = False
    use_alpha_as_mask = False
    mask_file = ""
    invert_mask = False
    mask_brightness_adjust = 1.0
    mask_contrast_adjust = 1.0
    overlay_mask = True
    mask_overlay_blur = 5

    mean_scale = 0
    var_scale = 0
    exposure_scale = 0
    exposure_target = 0.5

    colormatch_scale = 0
    colormatch_image = "https://www.saasdesign.io/wp-content/uploads/2021/02/palette-3-min-980x588.png"
    colormatch_n_colors = 4
    ignore_sat_weight = 0

    init_mse_scale = 0
    init_mse_image = ""
    blue_scale = 0

    gradient_wrt = 'x0_pred'
    gradient_add_to = 'both'
    decode_method = 'linear'
    grad_threshold_type = 'dynamic'
    clamp_grad_threshold = 0.2
    clamp_start = 0.2
    clamp_stop = 0.01
    grad_inject_timing = list(range(1, 10))

    cond_uncond_sync = True
    precision = 'autocast'
    C = 4
    f = 8

    cond_prompt = ""
    cond_prompts = ""
    uncond_prompt = ""
    uncond_prompts = ""
    timestring = ""
    init_latent = None
    init_sample = None
    init_sample_raw = None
    mask_sample = None
    init_c = None
    seed_internal = 0

    return locals()


# %%
# --------------------------------------------------------------------------- #
#  Run
# --------------------------------------------------------------------------- #
override_settings_with_file = False
settings_file = "custom"
custom_settings_file = ""

args_dict = DeforumArgs()
anim_args_dict = DeforumAnimArgs()

if override_settings_with_file:
    load_args(args_dict, anim_args_dict, settings_file, custom_settings_file, verbose=False)

args = SimpleNamespace(**args_dict)
anim_args = SimpleNamespace(**anim_args_dict)

args.timestring = time.strftime('%Y%m%d%H%M%S')
args.strength = max(0.0, min(1.0, args.strength))

if args.seed == -1:
    args.seed = random.randint(0, 2**32 - 1)
if not args.use_init:
    args.init_image = None

if anim_args.animation_mode == 'None':
    anim_args.max_frames = 1
elif anim_args.animation_mode == 'Video Input':
    args.use_init = True

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

cond, uncond = Prompts(prompt=prompts, neg_prompt=neg_prompts).as_dict()

print(f"\n{'='*60}")
print(f"  Deforum x FLUX.2-klein-4B")
print(f"  Mode: {anim_args.animation_mode} | Steps: {args.steps} | Scale: {args.scale}")
print(f"  Resolution: {args.W}x{args.H} | Frames: {anim_args.max_frames}")
print(f"  Output: {args.outdir}")
print(f"{'='*60}\n")

try:
    if anim_args.animation_mode in ('2D', '3D'):
        render_animation(root, anim_args, args, cond, uncond)
    elif anim_args.animation_mode == 'Video Input':
        render_input_video(root, anim_args, args, cond, uncond)
    elif anim_args.animation_mode == 'Interpolation':
        render_interpolation(root, anim_args, args, cond, uncond)
    else:
        render_image_batch(root, args, cond, uncond)
except Exception as e:
    print(f"\nError during rendering: {e}")
    import traceback
    traceback.print_exc()
finally:
    try:
        ffmpeg_args_dict = ffmpegArgs()
        ffmpeg_args = SimpleNamespace(**ffmpeg_args_dict)
        make_mp4_ffmpeg(ffmpeg_args, display_ffmpeg=False, debug=False)
    except Exception as e:
        print(f"FFmpeg export failed: {e}")

print("\nDone!")
