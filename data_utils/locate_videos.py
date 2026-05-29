import argparse
import json
from pathlib import Path

import logging

import imageio
from tqdm import tqdm

logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v", ".mpeg", ".mpg"}


def get_video_metadata(video_path: str) -> dict:
    try:
        reader = imageio.get_reader(video_path, "ffmpeg")
        meta = reader.get_meta_data()
        width, height = meta["size"]
        fps = round(float(meta["fps"]), 3)
        duration = round(float(meta["duration"]), 3)
        codec = meta.get("codec")
        pix_fmt = meta.get("pix_fmt")
        reader.close()
        return {"width": width, "height": height, "fps": fps, "duration": duration, "codec": codec, "pix_fmt": pix_fmt}
    except Exception as e:
        return {"width": None, "height": None, "fps": None, "duration": None, "codec": None, "pix_fmt": None, "error": f"{type(e).__name__}: {e}"}


def locate_videos(root_dir: str, output_file: str):
    root = Path(root_dir)
    video_paths = sorted(
        p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS
    )

    output_path = Path(output_file)
    if output_path.exists():
        output_path.unlink()

    with open(output_file, "w") as f:
        for path in tqdm(video_paths, desc="Processing videos"):
            resolved = str(path.resolve())
            meta = get_video_metadata(resolved)
            f.write(json.dumps({"video_path": resolved, **meta}) + "\n")

    print(f"Found {len(video_paths)} videos. Saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursively find video files and save paths to a JSONL file.")
    parser.add_argument("-r", "--root_dir", help="Root directory to search")
    args = parser.parse_args()

    locate_videos(args.root_dir, output_file=args.root_dir + "/metadata.jsonl")
