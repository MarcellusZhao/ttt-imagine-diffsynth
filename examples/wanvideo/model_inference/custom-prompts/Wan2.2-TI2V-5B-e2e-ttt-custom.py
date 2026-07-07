"""
End-to-End Test-Time Training (E2E-TTT) sequential long-video generation — Wan2.2-TI2V-5B.

For a single narrative prompt, the video is generated chunk by chunk: generate chunk k,
*memorize* it with in-place first-order LoRA updates, then generate chunk k+1 with the
adapted LoRA. The LoRA scratchpad is reset to the meta-init phi_0 before each narrative.

TI2V-5B can also condition the first chunk on an image: pass --input_image (the
remaining chunks continue purely from the adapted LoRA).

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
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt describing the video to generate.")
    parser.add_argument("--negative_prompt", type=str, required=True,
                        help="Negative prompt.")

    # LoRA "memory scratchpad"
    parser.add_argument("--lora", type=str, required=True,
                        help="Path to a meta-trained LoRA phi_0 checkpoint.")
    parser.add_argument("--algorithm", type=str, required=True, choices=["maml", "fomaml", "reptile"],
                        help="Meta-training algorithm.")
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank.")
    parser.add_argument("--target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2",
                        help="Comma-separated module name patterns to inject LoRA into.")

    # Inner-loop (memorization) config
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

    # TI2V-5B autoregressive anchoring
    parser.add_argument("--no_condition_on_last_frame", dest="condition_on_last_frame",
                        action="store_false",
                        help="Disable autoregressively anchoring each chunk on the previous chunk's "
                             "last frame (enabled by default).")
    parser.add_argument("--no_drop_boundary_frame", dest="drop_boundary_frame",
                        action="store_false",
                        help="Keep the duplicated anchor frame at each chunk boundary "
                             "(dropped by default).")

    # Output
    parser.add_argument("--output-dir", type=str, default=f"./results/custom-prompts",
                        help="Output directory.")
    parser.add_argument("--output_name", type=str, default="Wan2.2-TI2V-5B-e2e-ttt-fomaml-with-conditioning.mp4", help="Output video name.")
    parser.add_argument("--fps", type=int, default=16, help="Output video FPS.")
    parser.add_argument("--quality", type=int, default=5, help="Output video quality.")

    return parser.parse_args()


def main():
    args = parse_args()

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

    output_dir = os.path.join(args.output_dir, args.prompt[:30])
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, args.output_name)

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
    )

    generator = WanE2ETTTSequentialGenerator(pipe, inner_cfg, infer_cfg, phi0=phi0)

    generate_kwargs = dict(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
    )

    start_time = time.time()
    frames = generator.generate(**generate_kwargs)
    end_time = time.time()
    print(f"Time taken to generate video: {end_time - start_time} seconds")

    save_video(frames, output_path, fps=args.fps, quality=args.quality)
    conditioning = "with" if args.condition_on_last_frame else "without"
    print(f"Saved a {len(frames)}-frame E2E-TTT long video with {args.algorithm} algorithm {conditioning} conditioning to {output_path}.")


if __name__ == "__main__":
    main()
