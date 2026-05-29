"""Re-encode videos listed in raw_metadata.jsonl to a uniform target FPS.

Reads the video list from <dataset>/raw_metadata.jsonl (one JSON object per
line, must have a "video_path" field), re-encodes each to the target FPS, and
writes the results under <dataset>/curated_videos/, preserving the path
structure relative to <dataset>/raw_videos/.  The output folder is created if
it does not exist.

Usage:
    python -m data_utils.resample_fps \\
        --dataset_path path/to/dataset \\
        --target_fps 16

    # Also resize:
    python -m data_utils.resample_fps ... --width 1280 --height 720

    # Skip ffmpeg (just copy) for files whose FPS already matches:
    python -m data_utils.resample_fps ... --copy-matching
"""

import argparse
import json
import os
import shutil
import subprocess
from os import path as osp
from typing import List, Optional

from imageio_ffmpeg import get_ffmpeg_exe
from tqdm import tqdm

from data_utils.inspect_video import probe_video


def build_ffmpeg_cmd(
    input_path: str,
    output_path: str,
    target_fps: int,
    width: Optional[int] = None,
    height: Optional[int] = None,
    crf: int = 18,
    preset: str = "medium",
) -> List[str]:
    """Compose an ffmpeg command line for FPS resampling.

    `-r <fps>` placed AFTER `-i` is the *output* frame rate. ffmpeg drops or
    duplicates frames as needed (no temporal interpolation). `-crf 18` is
    visually-lossless x264; lower = higher quality (and larger files).
    """
    cmd = [get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", input_path]
    if width is not None and height is not None:
        cmd += ["-vf", f"scale={width}:{height}"]
    cmd += [
        "-r", str(target_fps),
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-pix_fmt", "yuv420p",
        "-an",  # drop audio: video-diffusion datasets are visual-only
        output_path,
    ]
    return cmd


def resample_one(
    input_path: str,
    output_path: str,
    target_fps: int,
    width: Optional[int] = None,
    height: Optional[int] = None,
    crf: int = 18,
    skip_if_exists: bool = True,
    copy_if_already_match: bool = False,
) -> str:
    """Resample one video. Returns a status string ('resampled', 'copied', ...).

    Raises on ffmpeg failure (caller decides whether to log-and-continue or abort).
    """
    if skip_if_exists and osp.exists(output_path):
        return "skipped (exists)"

    os.makedirs(osp.dirname(output_path), exist_ok=True)

    # Fast path: if input fps already matches and no resize requested, just copy.
    if copy_if_already_match and width is None and height is None:
        meta = probe_video(input_path)
        if meta.error is None and round(meta.fps, 3) == round(float(target_fps), 3):
            shutil.copy2(input_path, output_path)
            return "copied (fps already matches)"

    cmd = build_ffmpeg_cmd(input_path, output_path, target_fps, width, height, crf)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # Remove partial output so a re-run does not see stale corrupt data.
        if osp.exists(output_path):
            os.remove(output_path)
        raise RuntimeError(f"ffmpeg failed (exit {proc.returncode}): {proc.stderr.strip()}")

    return "resampled"


def collect_videos(root: str, recursive: bool, ext: str = ".mp4") -> List[str]:
    """List all `ext` files at `root`. Accepts either a file or a directory."""
    if osp.isfile(root):
        return [root]
    if not osp.isdir(root):
        raise FileNotFoundError(root)
    videos: List[str] = []
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.endswith(ext):
                    videos.append(osp.join(dirpath, f))
    else:
        videos = [
            osp.join(root, f) for f in os.listdir(root)
            if f.endswith(ext) and osp.isfile(osp.join(root, f))
        ]
    return sorted(videos)


def relative_output(video_path: str, input_root: str) -> str:
    """Path of `video_path` relative to `input_root`, or basename if root is a file."""
    if osp.isfile(input_root):
        return osp.basename(video_path)
    return osp.relpath(video_path, input_root)


def load_records_from_jsonl(jsonl_path: str) -> List[dict]:
    """Return all records from a JSONL metadata file that have a video_path."""
    records: List[dict] = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("video_path") or record.get("path"):
                records.append(record)
    return records


def make_curated_record(raw: dict, out_path: str, target_fps: int,
                        width: Optional[int], height: Optional[int]) -> dict:
    """Build a curated metadata record by copying raw fields and updating what changed."""
    record = dict(raw)
    # Update the path key that was present in the raw record.
    if "video_path" in record:
        record["video_path"] = out_path
    else:
        record["path"] = out_path
    record["fps"] = float(target_fps)
    record["codec"] = "h264"
    record["pix_fmt"] = "yuv420p"
    if width is not None and height is not None:
        record["width"] = width
        record["height"] = height
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset_path", required=True,
                        help="Dataset root folder; reads <dataset_path>/raw_metadata.jsonl "
                             "and writes to <dataset_path>/curated_videos/ and "
                             "<dataset_path>/curated_metadata.jsonl")
    parser.add_argument("--target_fps", type=int, required=True,
                        help="Output frame rate. Lower = drops frames; higher = duplicates frames.")
    parser.add_argument("--width", type=int, default=None,
                        help="Optional resize width (must be set with --height)")
    parser.add_argument("--height", type=int, default=None,
                        help="Optional resize height (must be set with --width)")
    parser.add_argument("--crf", type=int, default=18,
                        help="x264 CRF (lower = higher quality; 18 ≈ visually lossless)")
    parser.add_argument("--preset", default="medium",
                        help="x264 preset: ultrafast..veryslow (slower = better compression)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-encode even if the output file already exists")
    parser.add_argument("--copy-matching", action="store_true",
                        help="If a file's fps already matches --target_fps and no resize "
                             "is requested, copy it instead of re-encoding (lossless, fast)")
    args = parser.parse_args()

    if (args.width is None) != (args.height is None):
        parser.error("--width and --height must be specified together")

    metadata_path = osp.join(args.dataset_path, "raw_metadata.jsonl")
    if not osp.exists(metadata_path):
        raise FileNotFoundError(f"raw_metadata.jsonl not found: {metadata_path}")

    input_dir = osp.join(args.dataset_path, "raw_videos")
    output_dir = osp.join(args.dataset_path, "curated_videos")
    curated_metadata_path = osp.join(args.dataset_path, "curated_metadata.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    all_records = load_records_from_jsonl(metadata_path)
    records = [
        r for r in all_records
        if osp.exists(r.get("video_path") or r.get("path", ""))
    ]
    missing = len(all_records) - len(records)
    if missing:
        print(f"Warning: {missing} path(s) from raw_metadata.jsonl not found on disk — skipped")
    if not records:
        print("No accessible videos found in raw_metadata.jsonl")
        return

    print(f"Resampling {len(records)} video(s) -> {args.target_fps} fps")
    print(f"  Metadata: {metadata_path}")
    print(f"  Output:   {output_dir}")
    if args.width is not None:
        print(f"  Resizing to {args.width}x{args.height}")

    counts = {
        "resampled": 0,
        "skipped (exists)": 0,
        "copied (fps already matches)": 0,
        "failed": 0,
    }
    curated_records: List[dict] = []
    for raw in tqdm(records):
        video = raw.get("video_path") or raw.get("path", "")
        out_path = osp.join(output_dir, relative_output(video, input_dir))
        try:
            status = resample_one(
                input_path=video,
                output_path=out_path,
                target_fps=args.target_fps,
                width=args.width,
                height=args.height,
                crf=args.crf,
                skip_if_exists=not args.overwrite,
                copy_if_already_match=args.copy_matching,
            )
            counts[status] = counts.get(status, 0) + 1
            curated_records.append(
                make_curated_record(raw, out_path, args.target_fps, args.width, args.height)
            )
        except Exception as e:
            counts["failed"] += 1
            print(f"FAILED {video}: {e}")

    with open(curated_metadata_path, "w", encoding="utf-8") as fh:
        for record in curated_records:
            fh.write(json.dumps(record) + "\n")

    print("\nResult:")
    for status, count in counts.items():
        print(f"  {count:>5}  {status}")
    print(f"\nCurated metadata -> {curated_metadata_path} ({len(curated_records)} records)")


if __name__ == "__main__":
    main()
