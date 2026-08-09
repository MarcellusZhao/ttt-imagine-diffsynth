"""
Chunk-by-chunk long-video generation with E2E-TTT-style anchoring — Wan2.2-TI2V-5B
(base model, no LoRA, no test-time adaptation).

This is the ablation partner of `Wan2.2-TI2V-5B-chunk-by-chunk-custom.py`: same
chunk-by-chunk loop, but the inter-chunk conditioning is exactly the one
`Wan2.2-TI2V-5B-e2e-ttt-custom.py` uses, so the *only* thing that differs from the
E2E-TTT runs is the LoRA memory scratchpad.

Two anchors are pinned into each follow-up chunk's leading latents via TI2V-5B's fused
VAE conditioning:
  * a **wide local anchor** — the previous chunk's trailing window, handed forward as ONE
    contiguous clip (`anchor_frames`) so it encodes to a k-latent *block* that carries real
    velocity, not a motion-ambiguous single frame. `--num_anchor_latent_frames` (k,
    default 3);
  * a **global anchor** — the very first frame of the whole video (the "sink"), captured
    once after chunk 0 and reused unchanged, which gives every chunk a non-sliding
    reference to how the video started and counteracts long-horizon drift.

With the sink on, the anchor block is displaced to latent positions 1..k, so the handed-forward
window is 4k+1 pixel frames (k=3 -> 13) — one extra leading frame purely as VAE causal
context, whose latent the sink then overwrites. Those pinned frames are context, not new
content, so they are trimmed at each boundary (`--no_drop_boundary_frame` keeps them).

`--no_condition_on_sink_frame` drops the global anchor; `--no_condition_on_last_frame`
drops both and makes every chunk an independent text-to-video generation.

`--prompt` takes **any number of prompts** (or `--prompt_file`, one per line), all served by
a single pipeline load — do not call this script once per prompt from a shell loop, that
re-loads the 5B DiT + umt5-xxl + VAE every time. Each prompt writes to its own
`<output-dir>/<prompt[:30]>/<output_name>`; `--skip_existing` makes a re-submission resume.
"""

import argparse
import glob
import os
import time

import torch
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion.e2e_ttt import (
    anchor_overlap_pixel_frames, num_clean_latents, num_pinned_pixel_frames,
)


# DEFAULT_PROMPT = "两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。"
# DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
DEFAULT_MODEL_DIR = "/work/nlp/hzhao/checkpoints/wan/Wan2.2-TI2V-5B"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chunk-by-chunk long-video generation with Wan2.2-TI2V-5B (base model) "
                    "using the E2E-TTT anchoring strategy (wide local anchor + first-frame sink)."
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
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of denoising steps.")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="Classifier-free guidance scale.")
    parser.add_argument("--sigma_shift", type=float, default=5.0, help="Flow-matching sigma shift.")

    # Inter-chunk anchoring (mirrors Wan2.2-TI2V-5B-e2e-ttt-custom.py)
    parser.add_argument("--no_condition_on_last_frame", dest="condition_on_last_frame",
                        action="store_false",
                        help="Disable anchoring each chunk on the previous chunk's trailing "
                             "window (enabled by default). Turning this off also drops the "
                             "sink, making every chunk an independent text-to-video generation.")
    parser.add_argument("--no_condition_on_sink_frame", dest="condition_on_sink_frame",
                        action="store_false",
                        help="Disable the fixed global anchor -- the video's very first "
                             "generated frame -- pinned alongside the sliding local anchor "
                             "(enabled by default). Requires last-frame conditioning.")
    parser.add_argument("--no_drop_boundary_frame", dest="drop_boundary_frame",
                        action="store_false",
                        help="Keep the pinned anchor frames at each chunk boundary "
                             "(dropped by default).")
    parser.add_argument("--num_anchor_latent_frames", type=int, default=3,
                        help="Width k of the local anchor block in LATENT frames (default 3). "
                             "k>1 hands the previous chunk's trailing window forward as one "
                             "contiguous encode, so the model sees actual velocity instead of "
                             "a motion-ambiguous single frame. k=1 is the legacy single-frame "
                             "anchor.")

    # Output
    parser.add_argument("--output-dir", type=str, default=f"./results/custom-prompts",
                        help="Output directory.")
    parser.add_argument("--output_name", type=str, default=None,
                        help="Output video name. Defaults to a name describing the anchoring.")
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


def generate_long_video(pipe, args, prompt, k_anchor, anchor_window, condition_on_sink_frame):
    """One long video for `prompt`: `--num_chunks` chunks, each anchored on the previous
    chunk's trailing window (+ the fixed first-frame sink). Returns (frames, seconds)."""
    all_frames = []
    # Running local anchor: the previous chunk's trailing `anchor_window` frames, re-encoded
    # as ONE contiguous clip by WanVideoUnit_ImageEmbedderFused so the anchor latents land at
    # the positions they belong to. k=1 keeps the legacy single-frame `input_image` path.
    cond_image = None
    cond_frames = None
    # Fixed global anchor: the video's first raw frame, captured once after chunk 0.
    sink_image = None
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
        # Anchor this follow-up chunk on the previous chunk's tail.
        if k > 0 and args.condition_on_last_frame:
            if k_anchor > 1 and cond_frames is not None:
                # `anchor_frames` supersedes `input_image` inside the fused embedder.
                call_kwargs["anchor_frames"] = cond_frames
            elif cond_image is not None:
                call_kwargs["input_image"] = cond_image
        sink_active = (
            k > 0 and condition_on_sink_frame
            and args.condition_on_last_frame and sink_image is not None
        )
        if sink_active:
            call_kwargs["sink_image"] = sink_image

        start_time = time.time()
        frames = pipe(**call_kwargs)
        end_time = time.time()
        total_time += end_time - start_time
        print(f"Time taken to generate chunk {k + 1} of {args.num_chunks}: {end_time - start_time} seconds")

        # The leading frames of a conditioned follow-up chunk reproduce the pinned anchor(s)
        # rather than new content; drop them to avoid a duplicate-frame seam at the boundary.
        if k > 0 and args.condition_on_last_frame and args.drop_boundary_frame:
            emitted = frames[num_pinned_pixel_frames(num_clean_latents(k_anchor, sink_active)):]
        else:
            emitted = frames
        all_frames.extend(emitted)

        # Carry the local anchor forward: the trailing window for the k>1 block path, the
        # single last frame for the legacy k=1 path.
        if args.condition_on_last_frame and len(frames) > 0:
            cond_image = frames[-1]
            cond_frames = frames[-anchor_window:] if k_anchor > 1 else None
            if cond_frames is not None and len(cond_frames) < anchor_window:
                # Would silently encode to fewer than k anchor latents. Only reachable with a
                # chunk shorter than the anchor window, i.e. a misconfigured frames_per_chunk.
                raise ValueError(
                    f"chunk {k} produced {len(frames)} frames but the k={k_anchor} anchor "
                    f"block needs {anchor_window}; raise --frames_per_chunk or lower "
                    f"--num_anchor_latent_frames."
                )
        # Capture the video's first raw frame once, as the fixed sink anchor.
        if k == 0 and sink_image is None and len(frames) > 0:
            sink_image = frames[0]

        anchoring = ""
        if k > 0 and args.condition_on_last_frame:
            anchoring = f" [anchored on prev chunk (k={k_anchor})" \
                        + (" + first-frame sink]" if sink_active else "]")
        print(f"[chunk-by-chunk] generated chunk {k + 1}/{args.num_chunks} "
              f"({len(emitted)} frames){anchoring}")

    return all_frames, total_time


def main():
    args = parse_args()
    prompts = resolve_prompts(args)

    # The sink is a *second* fused clean frame pinned next to the sliding local anchor, so it
    # only exists on the last-frame conditioning path (same normalization as the E2E-TTT script).
    condition_on_sink_frame = args.condition_on_sink_frame and args.condition_on_last_frame
    if args.condition_on_sink_frame and not args.condition_on_last_frame:
        print("[chunk-by-chunk] NOTE: the first-frame sink requires last-frame conditioning; "
              "generating without the sink.")

    k_anchor = max(1, int(args.num_anchor_latent_frames))
    # Pixel frames of each chunk handed forward as the next chunk's anchor block, and the
    # number of leading decoded frames that are pinned context rather than new content.
    anchor_window = anchor_overlap_pixel_frames(k_anchor, condition_on_sink_frame)
    n_clean = num_clean_latents(k_anchor, condition_on_sink_frame)
    pinned = num_pinned_pixel_frames(n_clean)
    if args.condition_on_last_frame and args.frames_per_chunk <= max(anchor_window, pinned):
        raise ValueError(
            f"frames_per_chunk={args.frames_per_chunk} is too short for a k={k_anchor} anchor "
            f"block (needs {anchor_window} frames handed forward, {pinned} pinned per chunk); "
            f"raise --frames_per_chunk or lower --num_anchor_latent_frames."
        )

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

    if args.output_name is not None:
        output_name = args.output_name
    else:
        sink_tag = "-ffsink" if condition_on_sink_frame else ""
        output_name = f"Wan2.2-TI2V-5B-chunk-by-chunk-k{k_anchor}{sink_tag}.mp4"

    if args.condition_on_last_frame:
        print(f"[chunk-by-chunk] anchor block k={k_anchor} latents"
              + (" + first-frame sink" if condition_on_sink_frame else "")
              + f" | {n_clean} pinned latents | {pinned} pinned pixel frames/chunk"
              + f" -> {args.frames_per_chunk - pinned} new frames per chunk"
              + f" | {anchor_window} frames handed forward")
        conditioning = f"with k={k_anchor} last-chunk anchoring" \
            + (" + first-frame sink" if condition_on_sink_frame else "")
    else:
        conditioning = "without conditioning"

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
        all_frames, total_time = generate_long_video(
            pipe, args, prompt, k_anchor, anchor_window, condition_on_sink_frame)
        run_time += total_time
        num_generated += 1

        save_video(all_frames, output_path, fps=args.fps, quality=args.quality)
        print(f"Total time taken to generate video: {total_time} seconds")
        print(f"Saved a {len(all_frames)}-frame chunk-by-chunk video {conditioning} to {output_path}.")

    print(f"[chunk-by-chunk] generated {num_generated}/{len(prompts)} prompt(s) "
          f"in {run_time:.1f}s total.")


if __name__ == "__main__":
    main()
