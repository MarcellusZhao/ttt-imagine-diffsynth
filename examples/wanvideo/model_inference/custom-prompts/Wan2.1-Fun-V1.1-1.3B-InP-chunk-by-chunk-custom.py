"""
Chunk-by-chunk long-video generation — Wan2.1-Fun-V1.1-1.3B-InP (base model, no adaptation).

For each narrative prompt, the long video is produced by generating `--num_chunks`
contiguous sub-clips one after another and concatenating them.

This is the *plain* autoregressive baseline: each follow-up chunk is anchored on the
previous chunk's last frame through the checkpoint's stock single-frame I2V conditioning
(`input_image`) — one given frame, zeros in every remaining slot of `y`, exactly as in I2V
pretraining. Its ablation partner
`Wan2.1-Fun-V1.1-1.3B-InP-chunk-by-chunk-anchored-custom.py` widens that to the full
E2E-TTT i2v anchoring (an m-frame motion window plus reference padding), and the E2E-TTT
driver adds the LoRA scratchpad on top. Read the three as: no anchor → single-frame anchor
→ wide anchor → wide anchor + memory.

A single anchor frame is **motion-unidentifiable** — no velocity can be read off one frame —
so this arm carries appearance across a boundary but not direction. That is the gap
`--num_motion_frames 5` closes in the anchored arm.

`--no_condition_on_last_chunk` makes every chunk an independent generation (null image
conditioning, see the base driver's docstring), which is the no-continuity floor.

`--prompt` takes **any number of prompts** (or `--prompt_file`, one per line), all served by
a single pipeline load — do not call this script once per prompt from a shell loop, that
re-loads the DiT + umt5-xxl + VAE + CLIP every time. Each prompt writes to its own
`<output-dir>/<prompt[:30]>/` subdirectory; `--skip_existing` makes a re-submission resume.
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
        description="Chunk-by-chunk long-video generation with Wan2.1-Fun-V1.1-1.3B-InP "
                    "(base model, stock single-frame I2V continuity, no adaptation)."
    )

    # Model loading
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR,
                        help="Directory containing the Wan2.1-Fun-V1.1-1.3B-InP checkpoint files.")
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
    parser.add_argument("--input_image", type=str, nargs="+", default=None,
                        help="Optional start image(s) seeding chunk 0 (I2V); later chunks "
                             "continue from the previous chunk's last frame. Either ONE "
                             "shared by every prompt, or exactly one per prompt (SVI's "
                             "(frame.jpg, prompt) pairing).")

    # Chunk-by-chunk controls
    parser.add_argument("--num_chunks", type=int, default=4,
                        help="Number of contiguous sub-clips to generate and concatenate.")
    parser.add_argument("--frames_per_chunk", type=int, default=45,
                        help="Frames per chunk (4n+1, aligns with the Wan temporal VAE "
                             "compression). 45 is the meta-training chunk length of the "
                             "E2E-TTT arm, kept here so the arms share a geometry.")

    # Generation settings
    parser.add_argument("--seed", type=int, default=0,
                        help="Base random seed; chunk k uses seed + k.")
    parser.add_argument("--height", type=int, default=480, help="Output video height.")
    parser.add_argument("--width", type=int, default=832, help="Output video width.")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of denoising steps.")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="Classifier-free guidance scale.")
    parser.add_argument("--sigma_shift", type=float, default=5.0, help="Flow-matching sigma shift.")

    # Inter-chunk conditioning
    parser.add_argument("--no_condition_on_last_chunk", dest="condition_on_last_chunk",
                        action="store_false",
                        help="Disable anchoring each chunk on the previous chunk's last frame "
                             "(stock single-frame I2V continuity). Enabled by default; when "
                             "disabled, every chunk is an independent generation on null image "
                             "conditioning.")
    parser.add_argument("--no_drop_boundary_frame", dest="drop_boundary_frame",
                        action="store_false",
                        help="Keep the duplicated anchor frame at each chunk boundary "
                             "(dropped by default).")

    # Output
    parser.add_argument("--output-dir", type=str, default="./results/custom-prompts",
                        help="Output directory.")
    parser.add_argument("--output_name", type=str, default=None,
                        help="Output video name. Defaults to a name describing the resolution.")
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
    prepended to the cross-attention context -- off-distribution on every call, and silently so.
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


def generate_long_video(pipe, args, prompt, input_image):
    """One long video for `prompt`: `--num_chunks` chunks concatenated, each optionally
    anchored on the previous chunk's last frame. Returns (frames, seconds).

    Per-prompt state (the running anchor) lives here rather than in `main`, so it is
    re-initialised for every prompt instead of leaking from one narrative into the next.
    """
    all_frames = []
    cond_image = input_image
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
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            sigma_shift=args.sigma_shift,
        )
        # Anchor on the previous chunk's last frame (or the seed image on chunk 0).
        if cond_image is not None and (k == 0 or args.condition_on_last_chunk):
            call_kwargs["input_image"] = cond_image

        start_time = time.time()
        frames = pipe(**call_kwargs)
        end_time = time.time()
        total_time += end_time - start_time
        print(f"Time taken to generate chunk {k + 1} of {args.num_chunks}: "
              f"{end_time - start_time} seconds")

        # The first frame of a conditioned follow-up chunk reproduces the anchor frame rather
        # than new content; drop it to avoid a duplicate-frame seam at the chunk boundary.
        anchored = k > 0 and args.condition_on_last_chunk and cond_image is not None
        emitted = frames[1:] if (anchored and args.drop_boundary_frame) else frames
        all_frames.extend(emitted)

        # Carry the last frame forward as the next chunk's anchor.
        if args.condition_on_last_chunk and len(frames) > 0:
            cond_image = frames[-1]

        print(f"[chunk-by-chunk] generated chunk {k + 1}/{args.num_chunks} "
              f"({len(emitted)} frames)" + (" [anchored on prev chunk]" if anchored else ""))

    return all_frames, total_time


def main():
    args = parse_args()
    prompts = resolve_prompts(args)
    input_images = resolve_input_images(args, prompts)

    # Loaded once for the whole prompt list — the DiT + umt5-xxl + VAE + CLIP cost minutes to
    # read off disk, so nothing below this line may depend on a single prompt.
    load_start = time.time()
    pipe = load_pipeline(args.model_dir, args.device)
    print(f"[chunk-by-chunk] loaded the pipeline in {time.time() - load_start:.1f}s "
          f"for {len(prompts)} prompt(s)")

    output_name = args.output_name or \
        f"Wan2.1-Fun-V1.1-1.3B-InP-chunk-by-chunk-{args.height}p.mp4"
    if args.condition_on_last_chunk:
        new_per_chunk = args.frames_per_chunk - (1 if args.drop_boundary_frame else 0)
        print(f"[chunk-by-chunk] single-frame I2V anchor | 1 pinned pixel frame/chunk -> "
              f"{new_per_chunk} new frames per chunk")

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

        # Every prompt uses the same base seed, so a prompt sampled here is reproducible
        # whether it was run alone or as part of a list.
        all_frames, total_time = generate_long_video(pipe, args, prompt, input_images[p_i])
        run_time += total_time
        num_generated += 1

        save_video(all_frames, output_path, fps=args.fps, quality=args.quality)
        print(f"Total time taken to generate video: {total_time} seconds")
        conditioning = "with" if args.condition_on_last_chunk else "without"
        print(f"Saved a {len(all_frames)}-frame chunk-by-chunk video {conditioning} "
              f"conditioning to {output_path}.")

    print(f"[chunk-by-chunk] generated {num_generated}/{len(prompts)} prompt(s) "
          f"in {run_time:.1f}s total.")


if __name__ == "__main__":
    main()
