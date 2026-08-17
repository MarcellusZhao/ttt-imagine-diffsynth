"""
Single-pass long-video generation — Wan2.1-Fun-V1.1-1.3B-InP (base model, one `pipe()` call
per video).

The upper baseline for the chunk-by-chunk arms: the whole video is denoised as ONE sequence,
so temporal coherence costs no inter-chunk machinery — and O(seq^2) attention makes it the
arm that stops scaling first. At 480x832 the Wan2.1 VAE (8x spatial, 4x temporal) gives
30x52 = 1560 DiT tokens per latent frame, so a 965-frame video is 241 latents ~ 376k tokens.

CAVEAT specific to this checkpoint: Fun-V1.1-1.3B-InP is an image-conditioned DiT
(`has_image_input` + `require_vae_embedding`), so it has no text-only forward. Without
`--input_image` the pipeline feeds it the NULL conditioning (zero mask, zero latents) — the
model's own condition-drop regime, which it is trained for but which we have not measured.
This is *not* the same thing as a plain T2V model like Wan2.1-T2V-1.3B, so do not read this
arm as "the T2V base"; it is "this checkpoint with nothing given".

`--prompt` takes **any number of prompts** (or `--prompt_file`, one per line), all served by
a single pipeline load — do not call this script once per prompt from a shell loop, that
re-loads the DiT + umt5-xxl + VAE + CLIP every time. Each prompt writes to its own
`<output-dir>/<prompt[:30]>/Wan2.1-Fun-V1.1-1.3B-InP.mp4`; `--skip_existing` makes a
re-submission resume.
"""

import argparse
import os
import time

import torch
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


# DEFAULT_PROMPT = "两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。"
# DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
DEFAULT_MODEL_DIR = "/work/nlp/hzhao/checkpoints/wan/Wan2.1-Fun-V1.1-1.3B-InP"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-pass text-to-video / image-to-video inference with "
                    "Wan2.1-Fun-V1.1-1.3B-InP."
    )

    # Model loading
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR,
                        help="Directory containing the Wan2.1-Fun-V1.1-1.3B-InP checkpoint files.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to load the pipeline on.")

    # Prompts
    parser.add_argument("--prompt", type=str, nargs="+", default=None,
                        help="One or more text prompts; each produces its own video, all "
                             "from a single pipeline load.")
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="Path to a .txt with one prompt per line ('#' lines skipped). "
                             "Overrides --prompt when set.")
    parser.add_argument("--negative_prompt", type=str, required=True,
                        help="Negative prompt (applied to every prompt).")
    parser.add_argument("--input_image", type=str, nargs="+", default=None,
                        help="Optional start image(s): either ONE shared by every prompt, or "
                             "exactly one per prompt (SVI's (frame.jpg, prompt) pairing). "
                             "Without it this checkpoint runs on null image conditioning -- "
                             "see the module docstring.")

    # Generation settings
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--height", type=int, default=480, help="Output video height.")
    parser.add_argument("--width", type=int, default=832, help="Output video width.")
    parser.add_argument("--num_frames", type=int, default=965,
                        help="Number of frames to generate (4n+1). 965 matches 24 chunks of "
                             "the chunk-by-chunk arms at frames_per_chunk=45 / motion window "
                             "5 (stride 40): 45 + 23*40.")

    # Output
    parser.add_argument("--output-dir", type=str, default="./results/custom-prompts",
                        help="Output directory.")
    parser.add_argument("--fps", type=int, default=16, help="Output video FPS.")
    parser.add_argument("--quality", type=int, default=5, help="Output video quality.")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip a prompt whose output video already exists (resumable).")

    args = parser.parse_args()
    if not args.prompt and not args.prompt_file:
        parser.error("one of --prompt / --prompt_file is required")
    return args


def resolve_prompts(args):
    """The prompts to generate, in order: --prompt_file (one per line, blank and
    '#'-commented lines skipped) when given, else the --prompt values."""
    if args.prompt_file:
        with open(args.prompt_file, "r") as f:
            prompts = [line.strip() for line in f
                       if line.strip() and not line.lstrip().startswith("#")]
        if not prompts:
            raise ValueError(f"--prompt_file '{args.prompt_file}' contains no prompts.")
        return prompts
    return list(args.prompt)


def resolve_input_images(args, prompts):
    """A start image per prompt, as a list parallel to `prompts` (entries may be None).

    Accepts one image shared by every prompt, or exactly one per prompt -- SVI's toy_test
    layout, where each sample is a (frame.jpg, prompt.txt) pair. Anything else is a
    mis-pairing that would silently give some prompt the wrong reference frame, so it raises.
    """
    if not args.input_image:
        return [None] * len(prompts)
    paths = list(args.input_image)
    if len(paths) == 1:
        paths = paths * len(prompts)
    elif len(paths) != len(prompts):
        raise ValueError(
            f"--input_image got {len(paths)} path(s) for {len(prompts)} prompt(s); pass either "
            f"one image (shared by all prompts) or exactly one per prompt, in the same order."
        )
    return [Image.open(p).convert("RGB") for p in paths]


def load_pipeline(model_dir, device):
    """Wan2.1-Fun-V1.1-1.3B-InP: DiT + umt5-xxl + VAE + **CLIP**.

    The image encoder is not optional on this family. The checkpoint has
    `require_clip_embedding`, so leaving it out makes the DiT run with no image tokens
    prepended to the cross-attention context -- off-distribution on every call, and silently
    so (the same requirement the meta-training config spells out for `model_paths`).
    """
    return WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(path=os.path.join(model_dir, "diffusion_pytorch_model.safetensors")),
            ModelConfig(path=os.path.join(model_dir, "models_t5_umt5-xxl-enc-bf16.pth")),
            ModelConfig(path=os.path.join(model_dir, "Wan2.1_VAE.pth")),
            ModelConfig(path=os.path.join(
                model_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")),
        ],
        tokenizer_config=ModelConfig(path=os.path.join(model_dir, "google/umt5-xxl")),
    )


def main():
    args = parse_args()
    prompts = resolve_prompts(args)
    input_images = resolve_input_images(args, prompts)

    # Loaded once for the whole prompt list — the DiT + umt5-xxl + VAE + CLIP cost minutes to
    # read off disk, so nothing below this line may depend on a single prompt.
    load_start = time.time()
    pipe = load_pipeline(args.model_dir, args.device)
    print(f"[base] loaded the pipeline in {time.time() - load_start:.1f}s "
          f"for {len(prompts)} prompt(s)")

    total_time = 0.0
    num_generated = 0
    for p_i, prompt in enumerate(prompts):
        output_dir = os.path.join(args.output_dir, prompt[:30])
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "Wan2.1-Fun-V1.1-1.3B-InP.mp4")
        if args.skip_existing and os.path.exists(output_path):
            print(f"[{p_i + 1}/{len(prompts)}] exists, skipping: {output_path}")
            continue
        print(f"[{p_i + 1}/{len(prompts)}] Generating for prompt: {prompt[:60]}...")

        # Every prompt uses the same seed, so a prompt sampled here is reproducible whether it
        # was run alone or as part of a list.
        pipe_kwargs = dict(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            tiled=True,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
        )
        if input_images[p_i] is not None:
            pipe_kwargs["input_image"] = input_images[p_i]

        start_time = time.time()
        video = pipe(**pipe_kwargs)
        elapsed = time.time() - start_time
        total_time += elapsed
        num_generated += 1
        print(f"Time taken to generate video: {elapsed} seconds")
        save_video(video, output_path, fps=args.fps, quality=args.quality)
        print(f"Saved a {len(video)}-frame video to {output_path}.")

    print(f"[base] generated {num_generated}/{len(prompts)} prompt(s) in {total_time:.1f}s total.")


if __name__ == "__main__":
    main()
