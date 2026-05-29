"""
Generate captions for input videos using CogVLM2-Llama3-Caption.
Model: THUDM/cogvlm2-llama3-caption

Reads video paths from a curated_metadata.jsonl file (one JSON object per line,
each with a "video_path" key). Saves captions.jsonl alongside the input file.
Each output line: {"video_path": "...", "caption": "..."}

Usage:
    python generate_caption.py --input path/to/curated_metadata.jsonl
"""

import argparse
import io
import json
import os
import traceback

import numpy as np
import torch
from decord import VideoReader, bridge, cpu
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/work/nlp/hzhao/checkpoints/cogvlm2-llama3-caption"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_TYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    else torch.float16
)

def load_video(video_data: bytes, strategy: str = "chat") -> torch.Tensor:
    """
    Extract frames from raw video bytes.
    strategy='chat': 1 frame per second, up to 24 frames.
    strategy='base': 24 evenly-spaced frames from the first 60 seconds.
    """
    bridge.set_bridge("torch")
    num_frames = 24
    decord_vr = VideoReader(io.BytesIO(video_data), ctx=cpu(0))
    total_frames = len(decord_vr)

    if strategy == "base":
        clip_start_sec, clip_end_sec = 0, 60
        start_frame = int(clip_start_sec * decord_vr.get_avg_fps())
        end_frame = min(total_frames, int(clip_end_sec * decord_vr.get_avg_fps()))
        frame_id_list = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    else:  # chat
        timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
        timestamps = [t[0] for t in timestamps]
        max_second = round(max(timestamps)) + 1
        frame_id_list = []
        for second in range(max_second):
            closest = min(timestamps, key=lambda x: abs(x - second))
            frame_id_list.append(timestamps.index(closest))
            if len(frame_id_list) >= num_frames:
                break

    video_tensor = decord_vr.get_batch(frame_id_list)
    return video_tensor.permute(3, 0, 1, 2)  # (C, T, H, W)


def predict(
    model,
    tokenizer,
    prompt: str,
    video_data: bytes,
    temperature: float = 0.1,
    max_new_tokens: int = 2048,
) -> str:
    strategy = "chat"
    video = load_video(video_data, strategy=strategy)

    inputs = model.build_conversation_input_ids(
        tokenizer=tokenizer,
        query=prompt,
        images=[video],
        history=[],
        template_version=strategy,
    )
    model_inputs = {
        "input_ids": inputs["input_ids"].unsqueeze(0).to(DEVICE),
        "token_type_ids": inputs["token_type_ids"].unsqueeze(0).to(DEVICE),
        "attention_mask": inputs["attention_mask"].unsqueeze(0).to(DEVICE),
        "images": [[inputs["images"][0].to(DEVICE).to(TORCH_TYPE)]],
    }
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": 128002,
        "top_k": 1,
        "do_sample": False,
        "top_p": 0.1,
        "temperature": temperature,
    }
    with torch.no_grad():
        outputs = model.generate(**model_inputs, **gen_kwargs)
        outputs = outputs[:, model_inputs["input_ids"].shape[1]:]
        return tokenizer.decode(outputs[0], skip_special_tokens=True)


def read_video_paths(input_jsonl: str) -> list[str]:
    """Read video paths from a curated_metadata.jsonl file."""
    paths = []
    with open(input_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                paths.append(record["video_path"])
    return paths


def main():
    parser = argparse.ArgumentParser(description="Video captioning with CogVLM2-Llama3-Caption")

    parser.add_argument("--input", required=True,
                        help="Path to curated_metadata.jsonl listing video paths")
    parser.add_argument("--prompt", default="Please describe this video in detail.",
                        help="Caption prompt (default: 'Please describe this video in detail.')")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature (default: 0.1)")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Max tokens to generate (default: 2048)")
    parser.add_argument("--quant", type=int, choices=[4, 8], default=0,
                        help="Enable 4-bit or 8-bit quantization")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.join(os.path.dirname(input_path), "captions.jsonl")

    # Load model
    print(f"Loading model from {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    quant_kwargs = {}
    if args.quant == 4:
        quant_kwargs = {"load_in_4bit": True}
    elif args.quant == 8:
        quant_kwargs = {"load_in_8bit": True}

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=TORCH_TYPE,
        trust_remote_code=True,
        **quant_kwargs,
    ).eval()

    if not quant_kwargs:
        model = model.to(DEVICE)

    video_paths = read_video_paths(input_path)
    if not video_paths:
        print("No video paths found in input file.")
        return

    print(f"Found {len(video_paths)} videos. Writing captions to {output_path}\n")
    with open(output_path, "w") as out_f:
        for i, video_path in enumerate(video_paths, 1):
            print(f"[{i}/{len(video_paths)}] Processing: {video_path}")
            try:
                with open(video_path, "rb") as f:
                    video_data = f.read()
                caption = predict(
                    model, tokenizer, args.prompt, video_data,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                )
                print(f"  Caption: {caption}\n")
            except Exception as e:
                print(f"  Error: {type(e).__name__}: {e}")
                traceback.print_exc()
                print()
                caption = None

            record = {"video_path": video_path, "caption": caption}
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

    print(f"Done. Captions saved to {output_path}")


if __name__ == "__main__":
    main()
