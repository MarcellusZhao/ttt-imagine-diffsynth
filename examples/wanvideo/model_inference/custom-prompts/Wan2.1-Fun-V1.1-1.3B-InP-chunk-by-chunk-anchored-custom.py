"""
Chunk-by-chunk long-video generation with E2E-TTT-style i2v anchoring —
Wan2.1-Fun-V1.1-1.3B-InP (base model, no LoRA, no test-time adaptation).

This is the ablation partner of `Wan2.1-Fun-V1.1-1.3B-InP-e2e-ttt-custom.py`: the same
chunk loop and *exactly* the same inter-chunk conditioning, so the only thing that differs
from an E2E-TTT run is the LoRA memory scratchpad. Keep `--num_chunks`,
`--frames_per_chunk`, `--num_motion_frames`, `--ref_pad_num` and the resolution in sync with
the TTT arm you are reading it against.

The conditioning is the **i2v anchor route** (`diffsynth/diffusion/e2e_ttt.py`), ported from
Stable-Video-Infinity, which drives this checkpoint's 14B sibling the same way. It rides
entirely on the DiT's *pretrained* I2V path — `y` = 4 mask channels + 16 image-latent
channels, plus CLIP tokens on the cross-attention context — with two extras:

  * a **motion window** — the previous chunk's trailing `--num_motion_frames` (m) PIXEL
    frames laid at pixel positions 0..m-1 of the VAE input, instead of a single frame. The
    causal VAE turns m into `motion_latent_frames(m)` latents of real content (1->1, 5->2,
    9->3), and a two-latent window is what carries *velocity* across the boundary. A single
    anchor frame is motion-unidentifiable, which is the whole reason m=5 exists;
  * **reference padding** (`--ref_pad_num`) — what fills the remaining pixel slots of `y`:
    `0` = zeros (stock I2V), `-1` = the video's own first frame in every slot (SVI's
    "padding for ID consistency"), `n > 0` = the first n. This is the i2v route's analogue
    of the fused route's first-frame sink, and it is where long-horizon identity drift is
    fought.

The load-bearing detail is what does NOT change: **the mask stays at the pretrained
`[1, 0, 0, ...]` pattern**, marking only pixel frame 0 as given, for every m. All the extra
conditioning rides in the image-latent *content*, in regions the mask calls "not given" —
context the model may use, not frames it is told to reproduce. That is what keeps the
channel in-distribution, and why m costs nothing in mask-distribution terms.

Cost is paid in new content instead: the chunk's leading m decoded frames regenerate the
window it was conditioned on, so they are trimmed at each boundary
(`--no_drop_boundary_frame` keeps them) and each chunk contributes `frames_per_chunk - m`
new frames.

`--no_condition_on_last_chunk` drops the anchor, making every chunk an independent
generation on null image conditioning.

`--prompt` takes **any number of prompts** (or `--prompt_file`, one per line), all served by
a single pipeline load — do not call this script once per prompt from a shell loop, that
re-loads the DiT + umt5-xxl + VAE + CLIP every time. Each prompt writes to its own
`<output-dir>/<prompt[:30]>/<output_name>`; `--skip_existing` makes a re-submission resume.
"""

import argparse
import os
import time

import torch
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion.e2e_ttt import motion_latent_frames


# DEFAULT_PROMPT = "两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。"
# DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
DEFAULT_MODEL_DIR = "/work/nlp/hzhao/checkpoints/wan/Wan2.1-Fun-V1.1-1.3B-InP"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chunk-by-chunk long-video generation with Wan2.1-Fun-V1.1-1.3B-InP "
                    "(base model) using the E2E-TTT i2v anchoring (motion window + "
                    "reference padding)."
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
                        help="Optional start image(s): either ONE shared by every prompt, or "
                             "exactly one per prompt (SVI's (frame.jpg, prompt) pairing). Each "
                             "seeds its prompt's chunk 0 motion window (the image repeated m "
                             "times, as SVI's --repeat_first_clip does) AND becomes the fixed "
                             "reference frame for --ref_pad_num, exactly as in the E2E-TTT "
                             "generator. Without it the reference is the video's own first "
                             "generated frame, captured after chunk 0.")

    # Chunk-by-chunk controls
    parser.add_argument("--num_chunks", type=int, default=4,
                        help="Number of contiguous sub-clips to generate and concatenate.")
    parser.add_argument("--frames_per_chunk", type=int, default=45,
                        help="Frames per chunk (4n+1, aligns with the Wan temporal VAE "
                             "compression). 45 is the meta-training chunk length of the "
                             "E2E-TTT arm; with m=5 the stride is 40 new frames per chunk.")

    # Generation settings
    parser.add_argument("--seed", type=int, default=0,
                        help="Base random seed; chunk k uses seed + k.")
    parser.add_argument("--height", type=int, default=480, help="Output video height.")
    parser.add_argument("--width", type=int, default=832, help="Output video width.")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of denoising steps.")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="Classifier-free guidance scale.")
    parser.add_argument("--sigma_shift", type=float, default=5.0, help="Flow-matching sigma shift.")

    # i2v anchoring (mirrors Wan2.1-Fun-V1.1-1.3B-InP-e2e-ttt-custom.py)
    parser.add_argument("--no_condition_on_last_chunk", dest="condition_on_last_chunk",
                        action="store_false",
                        help="Disable anchoring each chunk on the previous chunk's trailing "
                             "motion window (enabled by default). With it off, every chunk is "
                             "an independent generation on null image conditioning.")
    parser.add_argument("--no_drop_boundary_frame", dest="drop_boundary_frame",
                        action="store_false",
                        help="Keep the m regenerated motion frames at each chunk boundary "
                             "(dropped by default).")
    parser.add_argument("--num_motion_frames", type=int, default=5,
                        help="Motion-window width m in PIXEL frames (SVI's --num_motion_frames). "
                             "1 reproduces stock single-frame I2V; 5 gives real content to the "
                             "first two latent frames, which is what carries velocity. Must "
                             "MATCH the E2E-TTT arm's --num_motion_frames. Also the chunk "
                             "overlap: each chunk contributes frames_per_chunk - m new frames.")
    parser.add_argument("--ref_pad_num", type=int, default=-1,
                        help="What fills the non-motion pixel slots of `y`: 0 = zeros, "
                             "-1 = the reference frame everywhere (SVI's ID-consistency "
                             "padding, the default), n > 0 = the first n.")

    # Output
    parser.add_argument("--output-dir", type=str, default="./results/custom-prompts",
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

    The image encoder is not optional on this family, and least of all on this route: the
    checkpoint has `require_clip_embedding`, and the i2v anchor feeds CLIP the motion
    window's first frame. Without it the DiT runs with no image tokens prepended to the
    cross-attention context -- off-distribution on every chunk, and silently so.
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


def generate_long_video(pipe, args, prompt, input_image, m):
    """One long video for `prompt`: `--num_chunks` chunks, each anchored on the previous
    chunk's trailing m-frame motion window (+ the fixed reference padding). Returns
    (frames, seconds).

    Deliberately a line-by-line mirror of the i2v branch of
    `WanE2ETTTSequentialGenerator.generate` -- same handoff, same chunk-0 regime, same trim.
    Per-prompt state (the running motion window and the reference frame) lives here, so it is
    re-initialised for every prompt instead of leaking from one narrative into the next.
    """
    all_frames = []
    # Running motion window: the previous chunk's trailing m frames, handed to the DiT's
    # pretrained conditioning path. On chunk 0 it is the seed image repeated m times, or None
    # (null conditioning) when no seed image was given.
    motion_frames = None
    # Fixed reference frame for --ref_pad_num. Known upfront when the caller supplied a start
    # image (so chunk 0 gets it too, as SVI passes its reference to every chunk); otherwise
    # captured once as the video's own first generated frame after chunk 0.
    ref_image = input_image
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
        if motion_frames is None and input_image is not None:
            motion_frames = [input_image] * m
        anchored = motion_frames is not None and (k == 0 or args.condition_on_last_chunk)
        if anchored:
            call_kwargs["anchor_frames"] = motion_frames
            call_kwargs["ref_pad_num"] = args.ref_pad_num
            if ref_image is not None:
                call_kwargs["sink_image"] = ref_image

        start_time = time.time()
        frames = pipe(**call_kwargs)
        end_time = time.time()
        total_time += end_time - start_time
        print(f"Time taken to generate chunk {k + 1} of {args.num_chunks}: "
              f"{end_time - start_time} seconds")

        # The chunk's leading m decoded frames regenerate the motion window it was conditioned
        # on, so they duplicate the previous chunk's tail. (SVI instead trims the *tail* of the
        # previous chunk; same count either way, but trimming the head keeps the original
        # frames rather than a regeneration of them.)
        trim = m if (k > 0 and args.condition_on_last_chunk and args.drop_boundary_frame) else 0
        emitted = frames[trim:]
        all_frames.extend(emitted)

        # Hand the trailing motion window forward.
        if len(frames) > 0:
            if len(frames) < m:
                raise ValueError(
                    f"chunk {k} produced {len(frames)} frames but the motion window needs "
                    f"{m}; raise --frames_per_chunk or lower --num_motion_frames."
                )
            motion_frames = frames[-m:]
        # Capture the video's first raw frame once, as the fixed reference.
        if k == 0 and ref_image is None and len(frames) > 0:
            ref_image = frames[0]

        anchoring = f" [anchored on prev chunk (m={m})]" \
            if (k > 0 and args.condition_on_last_chunk) else ""
        print(f"[chunk-by-chunk-anchored] generated chunk {k + 1}/{args.num_chunks} "
              f"({len(emitted)} frames){anchoring}")

    return all_frames, total_time


def main():
    args = parse_args()
    prompts = resolve_prompts(args)
    input_images = resolve_input_images(args, prompts)

    m = max(1, int(args.num_motion_frames))
    # The motion window is both what is handed forward and what is regenerated (and trimmed)
    # at the head of each follow-up chunk, so a chunk shorter than it makes no sense -- and
    # the VAE-encode path would raise from deep inside the pipeline rather than here.
    if args.condition_on_last_chunk and args.frames_per_chunk <= m:
        raise ValueError(
            f"frames_per_chunk={args.frames_per_chunk} is too short for an m={m} motion "
            f"window (each chunk would contribute {args.frames_per_chunk - m} new frames); "
            f"raise --frames_per_chunk or lower --num_motion_frames."
        )

    # Loaded once for the whole prompt list — the DiT + umt5-xxl + VAE + CLIP cost minutes to
    # read off disk, so nothing below this line may depend on a single prompt.
    load_start = time.time()
    pipe = load_pipeline(args.model_dir, args.device)
    print(f"[chunk-by-chunk-anchored] loaded the pipeline in {time.time() - load_start:.1f}s "
          f"for {len(prompts)} prompt(s)")

    if args.output_name is not None:
        output_name = args.output_name
    else:
        pad_tag = "" if args.ref_pad_num == 0 else f"-refpad{args.ref_pad_num}"
        output_name = (f"Wan2.1-Fun-V1.1-1.3B-InP-chunk-by-chunk-anchored"
                       f"-m{m}{pad_tag}-{args.height}p.mp4")

    if args.condition_on_last_chunk:
        pad = args.ref_pad_num
        pad_desc = "zeros" if pad == 0 else (
            "reference frame everywhere" if pad < 0 else f"reference frame x{pad}")
        trimmed = m if args.drop_boundary_frame else 0
        print(f"[chunk-by-chunk-anchored] anchoring: pretrained i2v (y + CLIP) | "
              f"motion window {m} pixel frames -> {motion_latent_frames(m)} latent frames | "
              f"ref_pad_num={pad} ({pad_desc}) | "
              f"mask left at the pretrained [1,0,0,...] pattern")
        print(f"[chunk-by-chunk-anchored] {trimmed} trimmed pixel frames/chunk -> "
              f"{args.frames_per_chunk - trimmed} new frames per chunk")
        conditioning = f"with m={m} motion-window anchoring (ref_pad_num={pad})"
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

        # Every prompt uses the same base seed, so a prompt sampled here is reproducible
        # whether it was run alone or as part of a list.
        all_frames, total_time = generate_long_video(pipe, args, prompt, input_images[p_i], m)
        run_time += total_time
        num_generated += 1

        save_video(all_frames, output_path, fps=args.fps, quality=args.quality)
        print(f"Total time taken to generate video: {total_time} seconds")
        print(f"Saved a {len(all_frames)}-frame chunk-by-chunk video {conditioning} "
              f"to {output_path}.")

    print(f"[chunk-by-chunk-anchored] generated {num_generated}/{len(prompts)} prompt(s) "
          f"in {run_time:.1f}s total.")


if __name__ == "__main__":
    main()
