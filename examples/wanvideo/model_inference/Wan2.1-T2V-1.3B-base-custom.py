import argparse
import os

import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


DEFAULT_PROMPT = "纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。"
DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
DEFAULT_MODEL_DIR = "/work/nlp/hzhao/checkpoints/wan/Wan2.1-T2V-1.3B"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Text-to-video / video-to-video inference with Wan2.1-T2V-1.3B."
    )

    # Model loading
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR,
                        help="Directory containing the Wan2.1-T2V-1.3B checkpoint files.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to load the pipeline on.")

    # Prompts
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT,
                        help="Text prompt describing the video to generate.")
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT,
                        help="Negative prompt.")

    # Generation settings
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--height", type=int, default=352, help="Output video height.")
    parser.add_argument("--width", type=int, default=512, help="Output video width.")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of denoising steps.")
    parser.add_argument("--num_frames", type=int, default=29,
                        help="Number of frames to generate.")

    # Output
    parser.add_argument("--output", type=str, default="video_1_Wan2.1-T2V-1.3B.mp4",
                        help="Output video file path.")
    parser.add_argument("--fps", type=int, default=15, help="Output video FPS.")
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

    pipe_kwargs = dict(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        tiled=True,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        num_frames=args.num_frames,
    )

    video = pipe(**pipe_kwargs)
    save_video(video, args.output, fps=args.fps, quality=args.quality)


if __name__ == "__main__":
    main()
