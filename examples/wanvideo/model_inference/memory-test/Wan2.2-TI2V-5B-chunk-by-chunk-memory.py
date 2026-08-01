"""
Chunk-by-chunk memory-test long-video generation — Wan2.2-TI2V-5B (base model, no adaptation).

The memory-test baseline for `Wan2.2-TI2V-5B-e2e-ttt-memory.py`: same per-chunk prompt
schedule, but plain chunk-by-chunk generation with NO LoRA memory scratchpad and NO
test-time adaptation. The only cross-chunk continuity is `--condition_on_last_chunk`
(TI2V-5B's native first-frame image conditioning on the previous chunk's last frame).

This is the control for the memory test: while the scripted character is out of frame,
the last-frame anchor holds a scene without it, so this baseline has no mechanism left to
carry the character's appearance across the gap. Comparing its reappearance against the
e2e-ttt runner's isolates what the LoRA memory contributes.

`--prompts` takes one sub-prompt per chunk (num_chunks = number of sub-prompts).
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
        description="Chunk-by-chunk memory-test long-video generation with Wan2.2-TI2V-5B "
                    "(base model, per-chunk prompt schedule, no adaptation)."
    )

    # Model loading
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR,
                        help="Directory containing the Wan2.2-TI2V-5B checkpoint files.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to load the pipeline on.")

    # Prompts — one sub-prompt per chunk (num_chunks is derived from the count).
    parser.add_argument("--prompts", type=str, nargs="+", required=True,
                        help="Per-chunk prompt schedule: one sub-prompt per chunk. The "
                             "number of chunks equals the number of sub-prompts.")
    parser.add_argument("--negative_prompt", type=str, required=True,
                        help="Negative prompt (applied to every chunk).")
    parser.add_argument("--num_chunks", type=int, default=None,
                        help="Number of chunks. Defaults to len(--prompts). Only needed to "
                             "broadcast a single sub-prompt across several chunks.")
    parser.add_argument("--case_name", type=str, default=None,
                        help="Name for the output subfolder. Defaults to the first "
                             "sub-prompt's first 30 chars.")

    # Chunk-by-chunk controls
    parser.add_argument("--frames_per_chunk", type=int, default=49,
                        help="Frames per chunk (4n+1, aligns with the Wan temporal VAE compression).")

    # Generation settings
    parser.add_argument("--seed", type=int, default=0,
                        help="Base random seed; chunk k uses seed + k.")
    parser.add_argument("--height", type=int, default=704, help="Output video height.")
    parser.add_argument("--width", type=int, default=1280, help="Output video width.")

    # Inter-chunk conditioning
    parser.add_argument("--no_condition_on_last_chunk", dest="condition_on_last_chunk",
                        action="store_false",
                        help="Disable anchoring each chunk on the previous chunk's last frame "
                             "(native I2V continuity). Enabled by default; when disabled, every "
                             "chunk is an independent text-to-video generation.")

    # Output
    parser.add_argument("--output-dir", type=str, default="./results/memory-test",
                        help="Output directory.")
    parser.add_argument("--output_name", type=str, default=None,
                        help="Output video name. Defaults to a name encoding the conditioning mode.")
    parser.add_argument("--fps", type=int, default=16, help="Output video FPS.")
    parser.add_argument("--quality", type=int, default=5, help="Output video quality.")

    return parser.parse_args()


def main():
    args = parse_args()

    num_chunks = args.num_chunks if args.num_chunks is not None else len(args.prompts)
    case_name = args.case_name if args.case_name is not None else args.prompts[0][:30]
    # Broadcast a single sub-prompt if the caller asked for more chunks than prompts.
    if len(args.prompts) == 1:
        prompts = args.prompts * num_chunks
    elif len(args.prompts) == num_chunks:
        prompts = args.prompts
    else:
        raise ValueError(
            f"per-chunk prompt schedule has {len(args.prompts)} entries but num_chunks="
            f"{num_chunks}; pass a single prompt or exactly num_chunks."
        )

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
    if args.output_name is not None:
        output_name = args.output_name
    elif args.condition_on_last_chunk:
        output_name = "Wan2.2-TI2V-5B-chunk-by-chunk-memory-with-conditioning.mp4"
    else:
        output_name = "Wan2.2-TI2V-5B-chunk-by-chunk-memory-without-conditioning.mp4"
    output_path = os.path.join(output_dir, output_name)

    print(f"[chunk-by-chunk memory] case '{case_name}': {num_chunks} chunks "
          f"({'with' if args.condition_on_last_chunk else 'without'} last-chunk conditioning)")
    for k, p in enumerate(prompts):
        print(f"  chunk {k}: {p}")

    all_frames = []
    cond_image = None
    total_time = 0
    for k in range(num_chunks):
        call_kwargs = dict(
            prompt=prompts[k],
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

        start_time = time.time()
        frames = pipe(**call_kwargs)
        end_time = time.time()
        total_time += end_time - start_time
        # The first frame of a conditioned follow-up chunk reproduces the anchor frame;
        # drop it to avoid a duplicate-frame seam at the chunk boundary.
        emitted = frames[1:] if (args.condition_on_last_chunk and k > 0) else frames
        all_frames.extend(emitted)
        if args.condition_on_last_chunk:
            cond_image = frames[-1]   # carry the last frame forward as the next chunk's anchor
        print(f"[chunk-by-chunk memory] generated chunk {k + 1}/{num_chunks} ({len(emitted)} frames)"
              + (" [anchored on prev chunk]" if args.condition_on_last_chunk and k > 0 else "")
              + f" | prompt: {prompts[k][:40]}...")

    save_video(all_frames, output_path, fps=args.fps, quality=args.quality)
    print(f"Total time taken to generate video: {total_time} seconds")
    conditioning = "with" if args.condition_on_last_chunk else "without"
    print(f"Saved a {len(all_frames)}-frame chunk-by-chunk memory-test video {conditioning} "
          f"conditioning to {output_path}.")


if __name__ == "__main__":
    main()
