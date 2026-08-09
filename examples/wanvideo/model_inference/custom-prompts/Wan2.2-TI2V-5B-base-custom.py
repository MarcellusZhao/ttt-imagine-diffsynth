"""
Single-pass long-video generation — Wan2.2-TI2V-5B (base model, one `pipe()` call per video).

`--prompt` takes **any number of prompts** (or `--prompt_file`, one per line), all served by
a single pipeline load — do not call this script once per prompt from a shell loop, that
re-loads the 5B DiT + umt5-xxl + VAE every time. Each prompt writes to its own
`<output-dir>/<prompt[:30]>/Wan2.2-TI2V-5B.mp4`; `--skip_existing` makes a re-submission
resume.
"""

import argparse
import glob
import os
import time

import torch
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


# DEFAULT_PROMPT = "两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。"
# DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
DEFAULT_MODEL_DIR = "/work/nlp/hzhao/checkpoints/wan/Wan2.2-TI2V-5B"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Text-to-video / image-to-video inference with Wan2.2-TI2V-5B."
    )

    # Model loading
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR,
                        help="Directory containing the Wan2.2-TI2V-5B checkpoint files.")
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

    # Generation settings
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--height", type=int, default=704, help="Output video height.")
    parser.add_argument("--width", type=int, default=1280, help="Output video width.")
    parser.add_argument("--num_frames", type=int, default=197,
                        help="Number of frames to generate.")

    # Output
    parser.add_argument("--output-dir", type=str, default=f"./results/custom-prompts",
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


def main():
    args = parse_args()
    prompts = resolve_prompts(args)

    # Loaded once for the whole prompt list — the 5B DiT + umt5-xxl + VAE cost minutes to
    # read off disk, so nothing below this line may depend on a single prompt.
    load_start = time.time()
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=[
            ModelConfig(path=os.path.join(args.model_dir, "models_t5_umt5-xxl-enc-bf16.pth")),
            ModelConfig(path=sorted(glob.glob(os.path.join(args.model_dir, "diffusion_pytorch_model*.safetensors")))),
            ModelConfig(path=os.path.join(args.model_dir, "Wan2.2_VAE.pth")),
        ],
        tokenizer_config=ModelConfig(path=os.path.join(args.model_dir, "google/umt5-xxl")),
    )
    print(f"[base] loaded the pipeline in {time.time() - load_start:.1f}s "
          f"for {len(prompts)} prompt(s)")

    total_time = 0.0
    num_generated = 0
    for p_i, prompt in enumerate(prompts):
        output_dir = os.path.join(args.output_dir, prompt[:30])
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"Wan2.2-TI2V-5B.mp4")
        if args.skip_existing and os.path.exists(output_path):
            print(f"[{p_i + 1}/{len(prompts)}] exists, skipping: {output_path}")
            continue
        print(f"[{p_i + 1}/{len(prompts)}] Generating for prompt: {prompt[:60]}...")

        # Every prompt uses the same seed, as it did when this script was invoked once per
        # prompt -- keeps previously sampled videos reproducible.
        pipe_kwargs = dict(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            tiled=True,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
        )

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
