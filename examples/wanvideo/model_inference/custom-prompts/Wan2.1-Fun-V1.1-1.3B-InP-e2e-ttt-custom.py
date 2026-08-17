"""
End-to-End Test-Time Training (E2E-TTT) sequential long-video generation —
Wan2.1-Fun-V1.1-1.3B-InP, **i2v anchor route**.

For each narrative prompt, the video is generated chunk by chunk: generate chunk k,
*memorize* it with in-place first-order LoRA updates, then generate chunk k+1 with the
adapted LoRA. The LoRA scratchpad is reset to the meta-init phi_0 before each narrative.

Two channels carry information across a chunk boundary, and they are complementary:

  * the **LoRA memory scratchpad** — meta-trained MAML-style so a few inner-loop steps on
    the chunks already generated generalize to the next one;
  * the **i2v anchor** — the DiT's *pretrained* I2V conditioning (`y` = 4 mask channels +
    16 image-latent channels, plus CLIP tokens on the cross-attention context), fed the
    previous chunk's trailing `--num_motion_frames` (m) PIXEL frames plus reference padding.
    This is the route Stable-Video-Infinity uses on this checkpoint's 14B sibling. Its
    no-adaptation ablation is `-chunk-by-chunk-anchored-custom.py`, which reproduces this
    conditioning exactly so the scratchpad is the only difference.

**Route note.** The anchoring flags here are `--num_motion_frames` / `--ref_pad_num`, NOT the
fused route's `--num_anchor_latent_frames` / `--no_condition_on_sink_frame`. m is in PIXEL
frames (the causal VAE turns it into `motion_latent_frames(m)` latents: 1->1, 5->2, 9->3),
and the i2v route's analogue of the fused route's first-frame sink is the reference padding:
`--ref_pad_num -1` fills every non-motion slot of `y` with the video's own first frame. The
fused sink is forced off on this route, on both the training and the sampling side.

**Everything below must match the phi_0 checkpoint.** A LoRA meta-init is not interchangeable
across anchor routes, and within this route it is not interchangeable across
`--optimizer` / `--inner_lr_init` / `--num_gradient_steps` / `--lora_rank` /
`--num_motion_frames` / `--ref_pad_num`. Note the effective test-time step count is
`ttt_steps_per_chunk x num_gradient_steps`, and that `--optimizer` defaults to `sgd` while
the meta-training config uses `adamw` — pass it explicitly.

`--prompt` takes **any number of prompts** (or `--prompt_file`, one per line), all served by
a single pipeline load — do not call this script once per prompt from a shell loop, that
re-loads the DiT + umt5-xxl + VAE + CLIP every time. Each prompt writes to its own
`<output-dir>/<prompt[:30]>/<output_name>`; `--skip_existing` makes a re-submission resume.

Point --lora at a meta-trained phi_0 (from
`model_training/lora/Wan2.1-Fun-V1.1-1.3B-InP-e2e-ttt-fomaml.sh`); a missing path falls back
to a zero-init identity adapter, i.e. the anchored baseline plus a no-op scratchpad.
"""

import argparse
import os
import time

import torch
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion.e2e_ttt import (
    InnerLoopConfig, InferenceConfig, inject_lora_for_ttt, WanE2ETTTSequentialGenerator,
    motion_latent_frames,
)


# DEFAULT_PROMPT = "两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。"
# DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
DEFAULT_MODEL_DIR = "/work/nlp/hzhao/checkpoints/wan/Wan2.1-Fun-V1.1-1.3B-InP"


def parse_args():
    parser = argparse.ArgumentParser(
        description="E2E-TTT sequential long-video generation with Wan2.1-Fun-V1.1-1.3B-InP "
                    "(i2v anchor route)."
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
                             "reference frame for --ref_pad_num. Without it chunk 0 gets the "
                             "NULL conditioning and the reference is the video's own first "
                             "generated frame -- which is what the shipped configs meta-train "
                             "for (e2e_ref_frame_source: first, e2e_p_motion_threshold: 1.0), "
                             "but is also a regime chunk 0 is never a predict target in.")

    # LoRA "memory scratchpad"
    parser.add_argument("--lora", type=str, required=True,
                        help="Path to a meta-trained LoRA phi_0 checkpoint.")
    parser.add_argument("--algorithm", type=str, required=True, choices=["maml", "fomaml", "reptile"],
                        help="Meta-training algorithm (recorded in the log line only; the "
                             "test-time update is first-order regardless).")
    parser.add_argument("--lora_rank", type=int, default=128,
                        help="LoRA rank. Must match the checkpoint's, or it will not load.")
    parser.add_argument("--target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2",
                        help="Comma-separated module name patterns to inject LoRA into. NOTE "
                             "this matches by suffix, so cross_attn.k_img / v_img and img_emb "
                             "are NOT adapted -- same as meta-training.")

    # Inner-loop (memorization) config
    parser.add_argument("--optimizer", type=str, default="sgd",
                        choices=["sgd", "adamw", "muon", "muonclip"],
                        help="Differentiable inner-loop optimizer for the memorization update. "
                             "The shipped meta-training config uses `adamw` -- pass it "
                             "explicitly, this default is `sgd`.")
    parser.add_argument("--num_gradient_steps", type=int, default=2,
                        help="Inner-loop gradient steps per memorization.")
    parser.add_argument("--num_mc_samples", type=int, default=1,
                        help="Monte-Carlo samples for the inner-loop loss.")
    parser.add_argument("--inner_lr_init", type=float, default=1e-5,
                        help="Initial inner-loop learning rate.")
    parser.add_argument("--max_inner_grad_norm", type=float, default=1.0,
                        help="Gradient-norm clip for the inner loop.")

    # Inference / chunking config
    parser.add_argument("--num_chunks", type=int, default=4,
                        help="Number of contiguous sub-clips to generate sequentially.")
    parser.add_argument("--frames_per_chunk", type=int, default=45,
                        help="Frames per chunk (4n+1, aligns with the Wan temporal VAE "
                             "compression). 45 is the meta-training chunk length; with m=5 "
                             "the stride is 40 new frames per chunk.")
    parser.add_argument("--ttt_steps_per_chunk", type=int, default=1,
                        help="Test-time training steps applied per chunk (the effective step "
                             "count is this x --num_gradient_steps).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--height", type=int, default=480, help="Output video height.")
    parser.add_argument("--width", type=int, default=832, help="Output video width.")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of denoising steps.")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="Classifier-free guidance scale.")
    parser.add_argument("--sigma_shift", type=float, default=5.0, help="Flow-matching sigma shift.")
    parser.add_argument("--no_gradient_checkpointing", dest="use_gradient_checkpointing",
                        action="store_false",
                        help="Retain all DiT activations during the test-time memorize "
                             "backward instead of recomputing them (enabled by default). "
                             "Checkpointing is numerically identical -- same update, one extra "
                             "forward per inner step against 50 sampling forwards -- and keeps "
                             "inner-loop memory from scaling with frames_per_chunk. Headroom "
                             "is comfortable at 1.3B/480p, so turning it off is a safe speedup "
                             "here; it is not on the 5B at 720p.")

    # i2v anchoring
    parser.add_argument("--no_condition_on_last_frame", dest="condition_on_last_frame",
                        action="store_false",
                        help="Disable anchoring each chunk on the previous chunk's trailing "
                             "motion window (enabled by default), leaving the LoRA scratchpad "
                             "as the only channel across a boundary.")
    parser.add_argument("--no_drop_boundary_frame", dest="drop_boundary_frame",
                        action="store_false",
                        help="Keep the m regenerated motion frames at each chunk boundary "
                             "(dropped by default).")
    parser.add_argument("--num_motion_frames", type=int, default=5,
                        help="Motion-window width m in PIXEL frames. Must MATCH the phi_0 "
                             "checkpoint's --e2e_num_motion_frames. 1 is stock single-frame "
                             "I2V and is motion-unidentifiable; 5 gives real content to the "
                             "first two latent frames, which is what carries velocity. Also "
                             "the chunk overlap: each chunk contributes fpc - m new frames.")
    parser.add_argument("--ref_pad_num", type=int, default=-1,
                        help="What fills the non-motion pixel slots of `y`: 0 = zeros, "
                             "-1 = the reference frame everywhere (SVI's ID-consistency "
                             "padding, the default), n > 0 = the first n. Must MATCH the "
                             "checkpoint's --e2e_ref_pad_num.")

    # Output
    parser.add_argument("--output-dir", type=str, default="./results/custom-prompts",
                        help="Output directory.")
    parser.add_argument("--output_name", type=str,
                        default="Wan2.1-Fun-V1.1-1.3B-InP-e2e-ttt.mp4",
                        help="Output video name.")
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
    cross-attention context -- off-distribution on every chunk, and silently so. This is the
    same requirement the meta-training config spells out for `model_paths`.
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

    m = max(1, int(args.num_motion_frames))
    # m is both the window handed forward and the head each follow-up chunk regenerates, so a
    # chunk no longer than it contributes nothing new. Raise here rather than from deep inside
    # the VAE-encode path.
    if args.condition_on_last_frame and args.frames_per_chunk <= m:
        raise ValueError(
            f"frames_per_chunk={args.frames_per_chunk} is too short for an m={m} motion "
            f"window (each chunk would contribute {args.frames_per_chunk - m} new frames); "
            f"raise --frames_per_chunk or lower --num_motion_frames."
        )

    # Loaded once for the whole prompt list — the DiT + umt5-xxl + VAE + CLIP cost minutes to
    # read off disk, so nothing below this line may depend on a single prompt.
    load_start = time.time()
    pipe = load_pipeline(args.model_dir, args.device)
    print(f"[E2E-TTT] loaded the pipeline in {time.time() - load_start:.1f}s "
          f"for {len(prompts)} prompt(s)")

    # Inject the LoRA "memory scratchpad" and (optionally) load the meta-trained phi_0.
    phi0 = inject_lora_for_ttt(
        pipe,
        lora_rank=args.lora_rank,
        target_modules=args.target_modules,
        lora_checkpoint=args.lora if (args.lora and os.path.exists(args.lora)) else None,
    )

    inner_cfg = InnerLoopConfig(
        num_gradient_steps=args.num_gradient_steps,
        num_mc_samples=args.num_mc_samples,
        inner_lr_init=args.inner_lr_init,
        max_inner_grad_norm=args.max_inner_grad_norm,
        optimizer=args.optimizer,
    )
    infer_cfg = InferenceConfig(
        num_chunks=args.num_chunks,
        frames_per_chunk=args.frames_per_chunk,   # 4n+1
        ttt_steps_per_chunk=args.ttt_steps_per_chunk,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        sigma_shift=args.sigma_shift,
        seed=args.seed,
        tiled=True,
        # ── i2v anchor route: the DiT's pretrained `y` + CLIP conditioning, SVI-style.
        i2v_anchor=True,
        num_motion_frames=m,                                  # PIXEL frames handed forward
        ref_pad_num=args.ref_pad_num,                         # fill for the non-motion slots
        condition_on_last_frame=args.condition_on_last_frame,
        drop_boundary_frame=args.drop_boundary_frame,         # trim the regenerated head
        # The fused route's first-frame sink does not exist here: this route's fixed global
        # reference IS the ref padding above. Meta-training forces --e2e_condition_on_sink_frame
        # off on the i2v route, so it is off here too. (The generator gates it on
        # `not i2v_anchor` anyway; set explicitly so the config reads as what it is.)
        condition_on_first_frame_sink=False,
    )
    print(f"[E2E-TTT] motion window {m} pixel frames -> {motion_latent_frames(m)} latent "
          f"frames of real content | {m if args.drop_boundary_frame else 0} trimmed pixel "
          f"frames/chunk -> {args.frames_per_chunk - (m if args.drop_boundary_frame else 0)} "
          f"new frames per chunk")

    # Built once as well: generate() restores the scratchpad to phi_0 before and after each
    # narrative, so looping over prompts here is exactly equivalent to one process per
    # prompt -- no adaptation bleeds from one video into the next.
    generator = WanE2ETTTSequentialGenerator(
        pipe, inner_cfg, infer_cfg, phi0=phi0,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
    )

    if args.condition_on_last_frame:
        pad = args.ref_pad_num
        conditioning = f"with m={m} motion-window anchoring" + (
            "" if pad == 0 else f" + reference padding (ref_pad_num={pad})")
    else:
        conditioning = "without conditioning"

    total_time = 0.0
    num_generated = 0
    for p_i, prompt in enumerate(prompts):
        output_dir = os.path.join(args.output_dir, prompt[:30])
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, args.output_name)
        if args.skip_existing and os.path.exists(output_path):
            print(f"[{p_i + 1}/{len(prompts)}] exists, skipping: {output_path}")
            continue
        print(f"[{p_i + 1}/{len(prompts)}] Generating for prompt: {prompt[:60]}...")

        # Every prompt uses the same base seed, so a prompt sampled here is reproducible
        # whether it was run alone or as part of a list.
        start_time = time.time()
        frames = generator.generate(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            input_image=input_images[p_i],
            seed=args.seed,
        )
        elapsed = time.time() - start_time
        total_time += elapsed
        num_generated += 1
        print(f"Time taken to generate video: {elapsed} seconds")

        save_video(frames, output_path, fps=args.fps, quality=args.quality)
        print(f"Saved a {len(frames)}-frame E2E-TTT long video with {args.algorithm} "
              f"algorithm {conditioning} to {output_path}.")

    print(f"[E2E-TTT] generated {num_generated}/{len(prompts)} prompt(s) in {total_time:.1f}s total.")


if __name__ == "__main__":
    main()
