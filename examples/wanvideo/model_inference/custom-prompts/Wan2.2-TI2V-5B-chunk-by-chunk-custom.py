"""
Chunk-by-chunk long-video generation — Wan2.2-TI2V-5B (base model, no adaptation).

For each narrative prompt, the long video is produced by generating
`--num_chunks` contiguous sub-clips one after another and concatenating them.

`--prompt` takes **any number of prompts** (or `--prompt_file`, one per line), all served by
a single pipeline load — do not call this script once per prompt from a shell loop, that
re-loads the 5B DiT + umt5-xxl + VAE every time. Each prompt writes to its own
`<output-dir>/<prompt[:30]>/` subdirectory; `--skip_existing` makes a re-submission resume.

`--condition_on_last_chunk` (on by default) anchors each follow-up chunk on the
previous chunk's last frame via TI2V-5B's native first-frame image conditioning,
so the long video stays temporally continuous. The duplicated anchor frame at each
chunk boundary is then dropped to avoid a seam. With the flag off, every chunk is
an independent text-to-video generation.

`--input_image` optionally seeds chunk 0 with an image (I2V); subsequent chunks
continue from the previous chunk's last frame.
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
        description="Chunk-by-chunk long-video generation with Wan2.2-TI2V-5B (base model, no adaptation)."
    )

    # Model loading
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR,
                        help="Directory containing the Wan2.2-TI2V-5B checkpoint files.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to load the pipeline on.")

    # Prompts
    parser.add_argument("--prompt", type=str, nargs="+", default=None,
                        help="One or more text prompts; each produces its own long video, "
                             "all from a single pipeline load.")
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="Path to a .txt with one prompt per line ('#' lines skipped). "
                             "Overrides --prompt when set.")
    parser.add_argument("--negative_prompt", type=str, required=True,
                        help="Negative prompt (applied to every prompt).")

    # Chunk-by-chunk controls
    parser.add_argument("--num_chunks", type=int, default=4,
                        help="Number of contiguous sub-clips to generate and concatenate.")
    parser.add_argument("--frames_per_chunk", type=int, default=49,
                        help="Frames per chunk (4n+1, aligns with the Wan temporal VAE compression).")

    # Generation settings
    parser.add_argument("--seed", type=int, default=0,
                        help="Base random seed; chunk k uses seed + k.")
    parser.add_argument("--height", type=int, default=704, help="Output video height.")
    parser.add_argument("--width", type=int, default=1280, help="Output video width.")

    # Inter-chunk conditioning
    parser.add_argument("--no_condition_on_last_chunk", dest="condition_on_last_chunk",
                        action="store_false",
                        help="Disable anchoring each chunk on the previous chunk's last frame "
                             "(native I2V continuity). Enabled by default; when disabled, every "
                             "chunk is an independent text-to-video generation.")

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


def generate_long_video(pipe, args, prompt):
    """One long video for `prompt`: `--num_chunks` chunks concatenated, each optionally
    anchored on the previous chunk's last frame. Returns (frames, seconds)."""
    all_frames = []
    cond_image = None
    total_time = 0
    for k in range(args.num_chunks):
        call_kwargs = dict(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            seed=args.seed + k,
            tiled=True,
            height=args.height,
            width=args.width,
            num_frames=args.frames_per_chunk,
        )
        # Anchor on the previous chunk's last frame (k > 0).
        if cond_image is not None:
            call_kwargs["input_image"] = cond_image

        start_time = time.time()
        frames = pipe(**call_kwargs)
        end_time = time.time()
        total_time += end_time - start_time
        print(f"Time taken to generate chunk {k + 1} of {args.num_chunks}: {end_time - start_time} seconds")
        # The first frame of a conditioned follow-up chunk reproduces the anchor frame;
        # drop it to avoid a duplicate-frame seam at the chunk boundary.
        emitted = frames[1:] if (args.condition_on_last_chunk and k > 0) else frames
        all_frames.extend(emitted)
        if args.condition_on_last_chunk:
            cond_image = frames[-1]   # carry the last frame forward as the next chunk's anchor
        print(f"[chunk-by-chunk] generated chunk {k + 1}/{args.num_chunks} ({len(emitted)} frames)"
              + (" [anchored on prev chunk]" if args.condition_on_last_chunk and k > 0 else ""))

    return all_frames, total_time


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
    print(f"[chunk-by-chunk] loaded the pipeline in {time.time() - load_start:.1f}s "
          f"for {len(prompts)} prompt(s)")

    conditioning = "with" if args.condition_on_last_chunk else "without"
    output_name = f"Wan2.2-TI2V-5B-chunk-by-chunk-{conditioning}-conditioning.mp4"

    run_time = 0.0
    num_generated = 0
    for p_i, prompt in enumerate(prompts):
        output_dir = os.path.join(args.output_dir, prompt[:30])
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_name)
        if args.skip_existing and os.path.exists(output_path):
            print(f"[{p_i + 1}/{len(prompts)}] exists, skipping: {output_path}")
            continue
        print(f"[{p_i + 1}/{len(prompts)}] Generating for prompt: {prompt[:60]}...")

        # Every prompt uses the same base seed, as it did when this script was invoked once
        # per prompt -- keeps previously sampled videos reproducible.
        all_frames, total_time = generate_long_video(pipe, args, prompt)
        run_time += total_time
        num_generated += 1

        save_video(all_frames, output_path, fps=args.fps, quality=args.quality)
        print(f"Total time taken to generate video: {total_time} seconds")
        print(f"Saved a {len(all_frames)}-frame chunk-by-chunk video {conditioning} conditioning to {output_path}.")

    print(f"[chunk-by-chunk] generated {num_generated}/{len(prompts)} prompt(s) "
          f"in {run_time:.1f}s total.")


if __name__ == "__main__":
    main()
