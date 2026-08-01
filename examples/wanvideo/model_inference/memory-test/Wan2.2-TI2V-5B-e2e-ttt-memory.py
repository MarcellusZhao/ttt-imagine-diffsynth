"""
E2E-TTT memory-test long-video generation — Wan2.2-TI2V-5B.

Same procedure as `custom-prompts/Wan2.2-TI2V-5B-e2e-ttt-custom.py` (generate chunk k,
*memorize* it with in-place first-order LoRA updates, generate chunk k+1 with the adapted
LoRA; the scratchpad is reset to phi_0 before the narrative) — the ONLY difference is that
the narrative is driven by a *per-chunk prompt schedule*: `--prompts` takes one sub-prompt
per chunk instead of a single prompt broadcast to every chunk.

Purpose — probe the LoRA "memory" module. Script a narrative in which a character leaves
the frame entirely for a few chunks and then returns:

    chunk 0-1: character A present alongside a fixed anchor B
    chunk 2-3: A has walked out of frame; only B remains
    chunk 4-5: A returns

While A is out of frame, the last-frame pixel anchor (`condition_on_last_frame`) holds a
scene *without* A, so nothing in pixel space can carry A's appearance across the gap —
only the LoRA scratchpad (phi_0 + the in-place TTT updates that memorised A earlier) can.
Run it with `--no_condition_on_last_frame` to attribute any preserved appearance purely to
the LoRA memory, and with conditioning on for the full deployed system; compare the two.

Point --lora at a meta-trained LoRA phi_0 checkpoint (from
`model_training/lora/Wan2.2-TI2V-5B-e2e-ttt-fomaml.sh`); a missing path falls back to a
zero-init identity adapter.
"""

import argparse
import glob
import os
import time

import torch
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion.e2e_ttt import (
    InnerLoopConfig, InferenceConfig, inject_lora_for_ttt,
    WanE2ETTTSequentialGenerator, restore_lora_state, ttt_update_inplace,
)


DEFAULT_MODEL_DIR = "/work/nlp/hzhao/checkpoints/wan/Wan2.2-TI2V-5B"


class MemoryTestSequentialGenerator(WanE2ETTTSequentialGenerator):
    """E2E-TTT sequential generator driven by a *per-chunk* prompt schedule.

    Identical to the base ``WanE2ETTTSequentialGenerator`` (per-narrative phi_0 reset,
    in-place first-order TTT memorise, optional last-frame anchoring) except that
    ``prompts[k]`` drives both the generation of chunk k and the memorise step that
    follows it. A single ``str`` (or length-1 list) is broadcast to every chunk, so this
    is a strict superset of the base behaviour. Kept here (not in the shared core) so
    ``diffsynth/diffusion/e2e_ttt.py`` stays untouched.
    """

    @staticmethod
    def _expand_prompts(prompt, num_chunks):
        if isinstance(prompt, str):
            return [prompt] * num_chunks
        prompts = list(prompt)
        if len(prompts) == 1:
            return prompts * num_chunks
        if len(prompts) != num_chunks:
            raise ValueError(
                f"per-chunk prompt schedule has {len(prompts)} entries but num_chunks="
                f"{num_chunks}; pass a single prompt, a length-1 list, or exactly num_chunks."
            )
        return prompts

    def generate(
        self,
        prompt,
        negative_prompt: str = "",
        *,
        input_image=None,
        seed=None,
        extra_call_kwargs=None,
    ):
        icfg = self.infer_cfg
        prompts = self._expand_prompts(prompt, icfg.num_chunks)
        seed = icfg.seed if seed is None else seed
        extra_call_kwargs = extra_call_kwargs or {}

        # Reset the memory scratchpad to phi_0 for this narrative.
        restore_lora_state(self.pipe.dit, self.phi0)

        all_frames = []
        cond_image = input_image
        for k in range(icfg.num_chunks):
            call_kwargs = dict(
                prompt=prompts[k],
                negative_prompt=negative_prompt,
                height=icfg.height,
                width=icfg.width,
                num_frames=icfg.frames_per_chunk,
                num_inference_steps=icfg.num_inference_steps,
                cfg_scale=icfg.cfg_scale,
                sigma_shift=icfg.sigma_shift,
                tiled=icfg.tiled,
                seed=seed + k,
            )
            if cond_image is not None and (k == 0 or icfg.condition_on_last_frame):
                call_kwargs["input_image"] = cond_image
            call_kwargs.update(extra_call_kwargs)

            # Latent handoff: memorise the sampler's final latents directly (no VAE
            # decode->re-encode round-trip), exactly as the base generator does.
            frames, chunk_latents = self.pipe(**call_kwargs, return_latents=True)
            if k > 0 and icfg.condition_on_last_frame and icfg.drop_boundary_frame:
                emitted = frames[1:]
            else:
                emitted = frames
            all_frames.extend(emitted)
            if icfg.condition_on_last_frame and len(frames) > 0:
                cond_image = frames[-1]
            print(f"[memory-test] generated chunk {k + 1}/{icfg.num_chunks} "
                  f"({len(emitted)} frames) | prompt: {prompts[k][:40]}...")

            if k == icfg.num_chunks - 1:
                continue

            # Memorise the chunk just generated with in-place first-order LoRA TTT.
            x0 = chunk_latents.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            context = self._encode_prompt(prompts[k])
            self.pipe.load_models_to_device(["dit"])
            for step in range(max(1, int(icfg.ttt_steps_per_chunk))):
                loss = ttt_update_inplace(
                    self.pipe, self.scheduler, x0, context, self.inner_cfg,
                    use_gradient_checkpointing=self.use_gradient_checkpointing,
                )
                print(f"[memory-test]  memorize chunk {k + 1} step {step + 1}/"
                      f"{icfg.ttt_steps_per_chunk} loss={loss:.6f}")

        # Leave the scratchpad at phi_0 for the next narrative.
        restore_lora_state(self.pipe.dit, self.phi0)
        return all_frames


def parse_args():
    parser = argparse.ArgumentParser(
        description="E2E-TTT memory-test long-video generation with Wan2.2-TI2V-5B "
                    "(per-chunk prompt schedule)."
    )

    # Model loading
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR,
                        help="Directory containing the Wan2.2-TI2V-5B checkpoint files.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to load the pipeline on.")

    # Prompts — one sub-prompt per chunk (num_chunks is derived from the count).
    parser.add_argument("--prompts", type=str, nargs="+", required=True,
                        help="Per-chunk prompt schedule: one sub-prompt per chunk. The "
                             "number of chunks equals the number of sub-prompts. A single "
                             "value is broadcast to every chunk (see --num_chunks).")
    parser.add_argument("--negative_prompt", type=str, required=True,
                        help="Negative prompt (applied to every chunk).")
    parser.add_argument("--num_chunks", type=int, default=None,
                        help="Number of chunks. Defaults to len(--prompts). Only needed to "
                             "broadcast a single sub-prompt across several chunks.")
    parser.add_argument("--case_name", type=str, default=None,
                        help="Name for the output subfolder. Defaults to the first "
                             "sub-prompt's first 30 chars (mirrors the custom-prompts scenario).")

    # LoRA "memory scratchpad"
    parser.add_argument("--lora", type=str, required=True,
                        help="Path to a meta-trained LoRA phi_0 checkpoint. A non-existent "
                             "path falls back to a zero-init (identity) adapter.")
    parser.add_argument("--algorithm", type=str, required=True, choices=["maml", "fomaml", "reptile"],
                        help="Meta-training algorithm that produced --lora (recorded for provenance).")
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
    parser.add_argument("--frames_per_chunk", type=int, default=49,
                        help="Frames per chunk (4n+1, aligns with the Wan temporal VAE compression).")
    parser.add_argument("--ttt_steps_per_chunk", type=int, default=1,
                        help="Test-time training steps applied per chunk.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed; chunk k uses seed + k.")
    parser.add_argument("--height", type=int, default=704, help="Output video height.")
    parser.add_argument("--width", type=int, default=1280, help="Output video width.")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of denoising steps.")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="Classifier-free guidance scale.")
    parser.add_argument("--sigma_shift", type=float, default=5.0, help="Flow-matching sigma shift.")

    # TI2V-5B autoregressive anchoring
    parser.add_argument("--no_condition_on_last_frame", dest="condition_on_last_frame",
                        action="store_false",
                        help="Disable autoregressively anchoring each chunk on the previous chunk's "
                             "last frame (enabled by default). Turn OFF to isolate the LoRA memory.")
    parser.add_argument("--no_drop_boundary_frame", dest="drop_boundary_frame",
                        action="store_false",
                        help="Keep the duplicated anchor frame at each chunk boundary "
                             "(dropped by default).")

    # Output
    parser.add_argument("--output-dir", type=str, default="./results/memory-test",
                        help="Output directory.")
    parser.add_argument("--output_name", type=str,
                        default="Wan2.2-TI2V-5B-e2e-ttt-memory.mp4", help="Output video name.")
    parser.add_argument("--fps", type=int, default=16, help="Output video FPS.")
    parser.add_argument("--quality", type=int, default=5, help="Output video quality.")

    return parser.parse_args()


def main():
    args = parse_args()

    num_chunks = args.num_chunks if args.num_chunks is not None else len(args.prompts)
    case_name = args.case_name if args.case_name is not None else args.prompts[0][:30]

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

    output_dir = os.path.join(args.output_dir, case_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, args.output_name)

    # Inject the LoRA "memory scratchpad" and (optionally) load the meta-trained phi_0.
    phi0 = inject_lora_for_ttt(
        pipe,
        lora_rank=args.lora_rank,
        target_modules=args.target_modules,
        lora_checkpoint=args.lora if (args.lora and os.path.exists(args.lora)) else None,
    )
    if not (args.lora and os.path.exists(args.lora)):
        print(f"[memory-test] WARNING: --lora '{args.lora}' not found; using a zero-init identity adapter.")

    inner_cfg = InnerLoopConfig(
        num_gradient_steps=args.num_gradient_steps,
        num_mc_samples=args.num_mc_samples,
        inner_lr_init=args.inner_lr_init,
        max_inner_grad_norm=args.max_inner_grad_norm,
        optimizer=args.optimizer,
    )
    infer_cfg = InferenceConfig(
        num_chunks=num_chunks,
        frames_per_chunk=args.frames_per_chunk,   # 4n+1
        ttt_steps_per_chunk=args.ttt_steps_per_chunk,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        sigma_shift=args.sigma_shift,
        seed=args.seed,
        tiled=True,
        condition_on_last_frame=args.condition_on_last_frame,
        drop_boundary_frame=args.drop_boundary_frame,
    )

    generator = MemoryTestSequentialGenerator(pipe, inner_cfg, infer_cfg, phi0=phi0)

    print(f"[memory-test/{args.algorithm}] case '{case_name}': {num_chunks} chunks "
          f"({'with' if args.condition_on_last_frame else 'without'} last-frame conditioning)")
    for k, p in enumerate(args.prompts):
        print(f"  chunk {k}: {p}")

    start_time = time.time()
    frames = generator.generate(
        prompt=args.prompts,
        negative_prompt=args.negative_prompt,
    )
    end_time = time.time()
    print(f"Time taken to generate video: {end_time - start_time} seconds")

    save_video(frames, output_path, fps=args.fps, quality=args.quality)
    conditioning = "with" if args.condition_on_last_frame else "without"
    print(f"Saved a {len(frames)}-frame E2E-TTT memory-test video with {args.algorithm} "
          f"algorithm {conditioning} conditioning to {output_path}.")


if __name__ == "__main__":
    main()
