"""
Base-model memory-test long-video generation — Wan2.2-TI2V-5B (no chunking, no adaptation).

The memory-test baseline that gives the base model the *whole* narrative at once: the same
per-chunk sub-prompts used by the e2e-ttt / chunk-by-chunk memory runners are concatenated
into a single long prompt, and one video is generated in a single pass (no chunk loop, no
LoRA memory, no test-time adaptation).

This is the "can the base model just do it from one rich prompt?" control for the memory
test. Comparing its appearance consistency across the leave/return arc against the chunked
runners shows how much the chunk-by-chunk pipeline (and, on top of it, the LoRA memory)
adds over plain long-form text-to-video.

`--prompts` takes the per-chunk sub-prompts (combined with `--joiner`); `--prompt` may be
passed instead to give an already-combined single prompt directly.
"""

import argparse
import glob
import os
import time

import torch
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


DEFAULT_MODEL_DIR = "/work/nlp/hzhao/checkpoints/wan/Wan2.2-TI2V-5B"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Base-model memory-test long-video generation with Wan2.2-TI2V-5B "
                    "(sub-prompts combined into one long prompt, single pass)."
    )

    # Model loading
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR,
                        help="Directory containing the Wan2.2-TI2V-5B checkpoint files.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to load the pipeline on.")

    # Prompts — the per-chunk sub-prompts, combined into one long prompt.
    parser.add_argument("--prompts", type=str, nargs="+", default=None,
                        help="Per-chunk sub-prompts to concatenate (with --joiner) into a "
                             "single long prompt. Mutually exclusive with --prompt.")
    parser.add_argument("--prompt", type=str, default=None,
                        help="An already-combined single long prompt (overrides --prompts).")
    parser.add_argument("--joiner", type=str, default="",
                        help="String inserted between sub-prompts when combining --prompts "
                             "(default: empty; the Chinese sub-prompts already end with a period).")
    parser.add_argument("--negative_prompt", type=str, required=True,
                        help="Negative prompt.")
    parser.add_argument("--case_name", type=str, default=None,
                        help="Name for the output subfolder. Defaults to the combined "
                             "prompt's first 30 chars.")

    # Generation settings
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--height", type=int, default=704, help="Output video height.")
    parser.add_argument("--width", type=int, default=1280, help="Output video width.")
    parser.add_argument("--num_frames", type=int, default=197,
                        help="Number of frames to generate in the single pass (4n+1).")

    # Output
    parser.add_argument("--output-dir", type=str, default="./results/memory-test",
                        help="Output directory.")
    parser.add_argument("--output_name", type=str,
                        default="Wan2.2-TI2V-5B-base-memory.mp4", help="Output video name.")
    parser.add_argument("--fps", type=int, default=16, help="Output video FPS.")
    parser.add_argument("--quality", type=int, default=5, help="Output video quality.")

    args = parser.parse_args()
    if args.prompt is None and not args.prompts:
        parser.error("provide either --prompt (a single combined prompt) or --prompts (sub-prompts).")
    return args


def main():
    args = parse_args()

    # Combine the per-chunk sub-prompts into one long prompt (unless a single --prompt
    # was given directly).
    if args.prompt is not None:
        prompt = args.prompt
    else:
        prompt = args.joiner.join(args.prompts)
    case_name = args.case_name if args.case_name is not None else prompt[:30]

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

    print(f"[base memory] case '{case_name}': {args.num_frames} frames, single pass")
    print(f"  combined prompt: {prompt}")

    pipe_kwargs = dict(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        tiled=True,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
    )

    start_time = time.time()
    video = pipe(**pipe_kwargs)
    end_time = time.time()
    print(f"Time taken to generate video: {end_time - start_time} seconds")
    save_video(video, output_path, fps=args.fps, quality=args.quality)
    print(f"Saved a {len(video)}-frame base-model memory-test video to {output_path}.")


if __name__ == "__main__":
    main()
