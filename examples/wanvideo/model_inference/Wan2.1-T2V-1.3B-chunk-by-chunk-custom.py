"""
Chunk-by-chunk long-video generation — Wan2.1-T2V-1.3B (base model, no adaptation).

This is the *baseline* counterpart to the E2E-TTT script
(`Wan2.1-T2V-1.3B-e2e-ttt.py`): for a single narrative prompt, the long video is
produced by generating `--num_chunks` contiguous sub-clips one after another and
concatenating them. Unlike E2E-TTT, the base model does NOT memorize the preceding
chunks and its weights are never modified — every chunk is an independent
text-to-video generation. It exists to show what long-video generation looks like
without any test-time training / LoRA "memory scratchpad".

Note: Wan2.1-T2V-1.3B is a pure T2V model with no native image conditioning, so this
uses the V2V seed-video path rather than true I2V `input_image` conditioning
(which is a no-op on this DiT). With the flag off, chunks are fully independent.
"""

import argparse
import os

import torch
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


# DEFAULT_PROMPT = "纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。"
# DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
DEFAULT_MODEL_DIR = "/work/nlp/hzhao/checkpoints/wan/Wan2.1-T2V-1.3B"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chunk-by-chunk long-video generation with Wan2.1-T2V-1.3B (base model, no adaptation)."
    )

    # Model loading
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR,
                        help="Directory containing the Wan2.1-T2V-1.3B checkpoint files.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to load the pipeline on.")

    # Prompts
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt describing the video to generate.")
    parser.add_argument("--negative_prompt", type=str, required=True,
                        help="Negative prompt.")

    # Chunk-by-chunk controls (mirror InferenceConfig in the E2E-TTT script).
    parser.add_argument("--num_chunks", type=int, default=2,
                        help="Number of contiguous sub-clips to generate and concatenate.")
    parser.add_argument("--frames_per_chunk", type=int, default=13,
                        help="Frames per chunk (4n+1, aligns with the Wan temporal VAE compression).")

    # Generation settings
    parser.add_argument("--seed", type=int, default=0,
                        help="Base random seed; chunk k uses seed + k.")
    parser.add_argument("--height", type=int, default=352, help="Output video height.")
    parser.add_argument("--width", type=int, default=512, help="Output video width.")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of denoising steps.")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="Classifier-free guidance scale.")
    parser.add_argument("--sigma_shift", type=float, default=5.0, help="Flow-matching sigma shift.")

    # Output
    parser.add_argument("--output-dir", type=str, default=f"./results/custom-prompts",
                        help="Output directory.")
    parser.add_argument("--fps", type=int, default=16, help="Output video FPS.")
    parser.add_argument("--quality", type=int, default=5, help="Output video quality.")
    return parser.parse_args()

def main():
    args = parse_args()
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=[
            ModelConfig(path=os.path.join(args.model_dir, "diffusion_pytorch_model.safetensors")),
            ModelConfig(path=os.path.join(args.model_dir, "models_t5_umt5-xxl-enc-bf16.pth")),
            ModelConfig(path=os.path.join(args.model_dir, "Wan2.1_VAE.pth")),
        ],
        tokenizer_config=ModelConfig(path=os.path.join(args.model_dir, "google/umt5-xxl")),
    )

    output_dir = os.path.join(args.output_dir, args.prompt[:30])
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"Wan2.1-T2V-1.3B-chunk-by-chunk.mp4")

    # Generate chunk by chunk. The base model never memorizes past chunks and its
    # weights are never modified. The only optional carry-over is the previous chunk
    # itself, fed back as a V2V seed when --condition_on_last_chunk is on.
    all_frames = []
    for k in range(args.num_chunks):
        call_kwargs = dict(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.frames_per_chunk,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            sigma_shift=args.sigma_shift,
            tiled=True,
            seed=args.seed + k,
        )

        frames = pipe(**call_kwargs)
        all_frames.extend(frames)
        print(f"[chunk-by-chunk] generated chunk {k + 1}/{args.num_chunks} ({len(frames)} frames)")

    save_video(all_frames, output_path, fps=args.fps, quality=args.quality)
    print(f"Saved a {len(all_frames)}-frame chunk-by-chunk long video to {output_path}.")


if __name__ == "__main__":
    main()
