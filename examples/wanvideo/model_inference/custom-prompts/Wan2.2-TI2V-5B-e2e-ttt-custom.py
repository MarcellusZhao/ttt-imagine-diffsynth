"""
End-to-End Test-Time Training (E2E-TTT) sequential long-video generation — Wan2.2-TI2V-5B.

For each narrative prompt, the video is generated chunk by chunk: generate chunk k,
*memorize* it with in-place first-order LoRA updates, then generate chunk k+1 with the
adapted LoRA. The LoRA scratchpad is reset to the meta-init phi_0 before each narrative.

`--prompt` takes **any number of prompts** (or `--prompt_file`, one per line), all served
by a single pipeline load — do not call this script once per prompt from a shell loop, that
re-loads the 5B DiT + umt5-xxl + VAE every time. Each prompt writes to its own
`<output-dir>/<prompt[:30]>/<output_name>`; `--skip_existing` makes a re-submission resume.

TI2V-5B can also condition the first chunk on an image: pass --input_image (the
remaining chunks continue purely from the adapted LoRA).

On top of the LoRA memory, each chunk is anchored on the previous chunk's last frame via
TI2V-5B's fused VAE first-frame latent, plus a *second*, fixed anchor: the very first
frame of the whole video (the "sink"), which gives every chunk a non-sliding reference to
how the video started and counteracts long-horizon drift. Both are on by default, matching
training; --no_condition_on_last_frame / --no_condition_on_sink_frame turn them off (drop
the sink when sampling a phi_0 meta-trained without --e2e_condition_on_sink_frame).

Point --lora at a meta-trained LoRA phi_0 checkpoint (from
`model_training/lora/Wan2.2-TI2V-5B-e2e-ttt-fomaml.sh`).
"""

import argparse
import glob
import os
import time

import torch
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion.e2e_ttt import (
    InnerLoopConfig, InferenceConfig, inject_lora_for_ttt, WanE2ETTTSequentialGenerator,
    num_clean_latents, num_pinned_pixel_frames, TTTDiagnostics,
)


# DEFAULT_PROMPT = "两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。"
# DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
DEFAULT_MODEL_DIR = "/work/nlp/hzhao/checkpoints/wan/Wan2.2-TI2V-5B"


def parse_args():
    parser = argparse.ArgumentParser(
        description="E2E-TTT sequential long-video generation with Wan2.2-TI2V-5B."
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

    # LoRA "memory scratchpad"
    parser.add_argument("--lora", type=str, required=True,
                        help="Path to a meta-trained LoRA phi_0 checkpoint.")
    parser.add_argument("--algorithm", type=str, required=True, choices=["maml", "fomaml", "reptile"],
                        help="Meta-training algorithm.")
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank.")
    parser.add_argument("--target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2",
                        help="Comma-separated module name patterns to inject LoRA into.")

    # Inner-loop (memorization) config
    parser.add_argument("--optimizer", type=str, default="sgd",
                        choices=["sgd", "adamw", "muon", "muonclip"],
                        help="Differentiable inner-loop optimizer for the memorization update.")
    parser.add_argument("--num_gradient_steps", type=int, default=1,
                        help="Inner-loop gradient steps per memorization.")
    parser.add_argument("--num_mc_samples", type=int, default=1,
                        help="Monte-Carlo samples for the inner-loop loss.")
    parser.add_argument("--inner_lr_init", type=float, default=1e-4,
                        help="Initial inner-loop learning rate.")
    parser.add_argument("--max_inner_grad_norm", type=float, default=1.0,
                        help="Gradient-norm clip for the inner loop.")

    # Inference / chunking config
    parser.add_argument("--num_chunks", type=int, default=4,
                        help="Number of contiguous sub-clips to generate sequentially.")
    parser.add_argument("--frames_per_chunk", type=int, default=49,
                        help="Frames per chunk (4n+1, aligns with the Wan temporal VAE compression).")
    parser.add_argument("--ttt_steps_per_chunk", type=int, default=1,
                        help="Test-time training steps applied per chunk.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--height", type=int, default=704, help="Output video height.")
    parser.add_argument("--width", type=int, default=1280, help="Output video width.")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of denoising steps.")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="Classifier-free guidance scale.")
    parser.add_argument("--sigma_shift", type=float, default=5.0, help="Flow-matching sigma shift.")
    parser.add_argument("--no_gradient_checkpointing", dest="use_gradient_checkpointing",
                        action="store_false",
                        help="Retain all DiT activations during the test-time memorize "
                             "backward instead of recomputing them (enabled by default). "
                             "Checkpointing is numerically identical -- same update, one extra "
                             "forward per inner step -- and is what keeps the inner loop inside "
                             "80-96GB at 720p; without it a 704x1280 chunk at "
                             "frames_per_chunk=53 OOMs on an H100.")

    # TI2V-5B autoregressive anchoring
    parser.add_argument("--no_condition_on_last_frame", dest="condition_on_last_frame",
                        action="store_false",
                        help="Disable autoregressively anchoring each chunk on the previous chunk's "
                             "last frame (enabled by default).")
    parser.add_argument("--no_drop_boundary_frame", dest="drop_boundary_frame",
                        action="store_false",
                        help="Keep the duplicated anchor frame at each chunk boundary "
                             "(dropped by default).")
    parser.add_argument("--no_condition_on_sink_frame", dest="condition_on_sink_frame",
                        action="store_false",
                        help="Disable additionally anchoring each follow-up chunk on the "
                             "video's very first generated frame (a fixed 'sink', enabled by "
                             "default alongside the sliding last-frame anchor). The sink needs "
                             "last-frame conditioning, and a phi_0 meta-trained with "
                             "--e2e_condition_on_sink_frame.")
    parser.add_argument("--num_anchor_latent_frames", type=int, default=1,
                        help="Width k of the local anchor block in LATENT frames (default 1 = "
                             "legacy single-frame anchor). Must MATCH the phi_0 checkpoint's "
                             "--e2e_num_anchor_latent_frames. k>1 hands the previous chunk's "
                             "trailing window forward as one contiguous encode, so the model "
                             "sees actual velocity instead of a motion-ambiguous single frame "
                             "-- the fix for next chunks that reverse the preceding motion.")

    # Inner-loop diagnostics (off by default; see TTTDiagnostics in diffsynth/diffusion/e2e_ttt.py)
    parser.add_argument("--ttt_diagnostics", type=str, default=None,
                        help="Write a per-step JSONL trace of the test-time inner loop to this "
                             "path: gradient norms before/after clipping, the displacement each "
                             "optimizer step actually realized, cumulative ||phi - phi_0|| split "
                             "over lora_A/lora_B, the effective ||B@A||/||W_base||, and two "
                             "FIXED-timestep memorize probes per chunk (against the chunk just "
                             "generated, and against chunk 0 held fixed for the whole video). "
                             "For diagnosing quality falling off with chunk index -- the loss "
                             "printed per chunk is a single random-sigma draw and cannot be read "
                             "as a trend. Costs ~4 extra forwards per chunk; the sampling RNG "
                             "stream is untouched, so clips stay byte-identical either way.")
    parser.add_argument("--ttt_probe_seeds", type=int, default=2,
                        help="Fixed (timestep, noise) draws averaged per probe. Only read when "
                             "--ttt_diagnostics is set. 0 disables the probes and keeps the "
                             "cheap per-step magnitudes.")

    # Output
    parser.add_argument("--output-dir", type=str, default=f"./results/custom-prompts",
                        help="Output directory.")
    parser.add_argument("--output_name", type=str, default="Wan2.2-TI2V-5B-e2e-ttt-fomaml-with-conditioning.mp4", help="Output video name.")
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

    # The sink is a *second* fused clean frame pinned next to the sliding last-frame
    # anchor, so it only exists on the last-frame conditioning path. Both default to on,
    # so --no_condition_on_last_frame alone implies no sink either (same normalization as
    # train_e2e_ttt.py).
    condition_on_sink_frame = args.condition_on_sink_frame and args.condition_on_last_frame
    if args.condition_on_sink_frame and not args.condition_on_last_frame:
        print("[E2E-TTT] NOTE: the first-frame sink requires last-frame conditioning; "
              "generating without the sink.")

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
        condition_on_last_frame=args.condition_on_last_frame,   # autoregressively anchor each chunk on the previous chunk's last frame
        drop_boundary_frame=args.drop_boundary_frame,           # drop the duplicated anchor frame at each chunk boundary
        condition_on_first_frame_sink=condition_on_sink_frame,   # also anchor on the video's first frame
        num_anchor_latent_frames=args.num_anchor_latent_frames,  # k: width of the local anchor block
    )
    _k = max(1, int(args.num_anchor_latent_frames))
    _n_clean = num_clean_latents(_k, condition_on_sink_frame)
    _pinned = num_pinned_pixel_frames(_n_clean)
    print(f"[E2E-TTT] anchor block k={_k} latents | {_n_clean} pinned latents | "
          f"{_pinned} pinned pixel frames/chunk -> {args.frames_per_chunk - _pinned} new frames per chunk")

    # Built once as well: generate() restores the scratchpad to phi_0 before and after each
    # narrative, so looping over prompts here is exactly equivalent to one process per
    # prompt -- no adaptation bleeds from one video into the next.
    # Owned here rather than inside the generator: the trace spans every prompt of this
    # invocation, and `generate` re-attaches per narrative so each video's drift is measured
    # from its own phi_0.
    diagnostics = TTTDiagnostics(
        args.ttt_diagnostics, num_probe_seeds=args.ttt_probe_seeds
    ) if args.ttt_diagnostics else None

    generator = WanE2ETTTSequentialGenerator(
        pipe, inner_cfg, infer_cfg, phi0=phi0,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        diagnostics=diagnostics,
    )

    if args.condition_on_last_frame:
        conditioning = "with last-frame + sink conditioning" if condition_on_sink_frame \
            else "with last-frame conditioning"
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

        # Every prompt uses the same base seed, as it did when this script was invoked once
        # per prompt -- keeps previously sampled videos reproducible.
        start_time = time.time()
        frames = generator.generate(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
        )
        elapsed = time.time() - start_time
        total_time += elapsed
        num_generated += 1
        print(f"Time taken to generate video: {elapsed} seconds")

        save_video(frames, output_path, fps=args.fps, quality=args.quality)
        print(f"Saved a {len(frames)}-frame E2E-TTT long video with {args.algorithm} algorithm {conditioning} to {output_path}.")

    if diagnostics is not None:
        diagnostics.close()
        print(f"[E2E-TTT] inner-loop trace written to {args.ttt_diagnostics}; summarize with:\n"
              f"  python eval/ttt_diagnostics/summarize.py {args.ttt_diagnostics}")

    print(f"[E2E-TTT] generated {num_generated}/{len(prompts)} prompt(s) in {total_time:.1f}s total.")


if __name__ == "__main__":
    main()
