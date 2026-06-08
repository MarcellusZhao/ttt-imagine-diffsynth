"""
Chunk-by-chunk long-video generation — Wan2.2-TI2V-5B (base model, no adaptation).

For a single narrative prompt, the long video is produced by generating
`--num_chunks` contiguous sub-clips one after another and concatenating them.

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

import torch
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


DEFAULT_PROMPT = "两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。"
DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
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
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT,
                        help="Text prompt describing the video to generate.")
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT,
                        help="Negative prompt.")

    # Chunk-by-chunk controls
    parser.add_argument("--num_chunks", type=int, default=4,
                        help="Number of contiguous sub-clips to generate and concatenate.")
    parser.add_argument("--frames_per_chunk", type=int, default=49,
                        help="Frames per chunk (4n+1, aligns with the Wan temporal VAE compression).")

    # Generation settings
    parser.add_argument("--seed", type=int, default=0,
                        help="Base random seed; chunk k uses seed + k.")
    parser.add_argument("--height", type=int, default=704, help="Output video height.")
    parser.add_argument("--width", type=int, default=1248, help="Output video width.")

    # Inter-chunk conditioning
    parser.add_argument("--no_condition_on_last_chunk", dest="condition_on_last_chunk",
                        action="store_false",
                        help="Disable anchoring each chunk on the previous chunk's last frame "
                             "(native I2V continuity). Enabled by default; when disabled, every "
                             "chunk is an independent text-to-video generation.")

    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Output video file path. If omitted, a descriptive name is auto-generated.")
    parser.add_argument("--fps", type=int, default=15, help="Output video FPS.")
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

    all_frames = []
    cond_image = None
    for k in range(args.num_chunks):
        call_kwargs = dict(
            prompt=args.prompt,
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

        frames = pipe(**call_kwargs)
        # The first frame of a conditioned follow-up chunk reproduces the anchor frame;
        # drop it to avoid a duplicate-frame seam at the chunk boundary.
        emitted = frames[1:] if (args.condition_on_last_chunk and k > 0) else frames
        all_frames.extend(emitted)
        if args.condition_on_last_chunk:
            cond_image = frames[-1]   # carry the last frame forward as the next chunk's anchor
        print(f"[chunk-by-chunk] generated chunk {k + 1}/{args.num_chunks} ({len(emitted)} frames)"
              + (" [anchored on prev chunk]" if args.condition_on_last_chunk and k > 0 else ""))

    if args.output is not None:
        output_path = args.output
    elif args.condition_on_last_chunk:
        output_path = (f"video_with_conditioning_chunk_by_chunk_Wan2.2-TI2V-5B"
                       f"_num_chunks_{args.num_chunks}_frames_per_chunk_{args.frames_per_chunk}.mp4")
    else:
        output_path = (f"video_without_conditioning_chunk_by_chunk_Wan2.2-TI2V-5B"
                       f"_num_chunks_{args.num_chunks}_frames_per_chunk_{args.frames_per_chunk}.mp4")

    save_video(all_frames, output_path, fps=args.fps, quality=args.quality)
    conditioning = "with" if args.condition_on_last_chunk else "without"
    print(f"Saved a {len(all_frames)}-frame chunk-by-chunk video {conditioning} conditioning to {output_path}.")


if __name__ == "__main__":
    main()
