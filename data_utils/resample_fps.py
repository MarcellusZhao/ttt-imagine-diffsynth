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

Encoding runs in parallel across files (one ffmpeg process per worker). On a
many-core box this is the dominant speedup since a single x264 encode can't use
all cores. Tune with --workers / --threads (workers x threads ~ core count), and
drop --preset to veryfast for a large per-file speedup at a small file-size cost:

    python -m data_utils.resample_fps ... --workers 32 --threads 4 --preset veryfast
"""

import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from os import path as osp
from typing import List, Optional

from imageio_ffmpeg import get_ffmpeg_exe
from tqdm import tqdm

from dataclasses import dataclass
import imageio

@dataclass
class VideoMetadata:
    path: str
    width: int
    height: int
    fps: float
    duration_sec: float
    num_frames: int
    codec: Optional[str] = None
    pix_fmt: Optional[str] = None
    file_size_mb: float = 0.0
    error: Optional[str] = None  # populated if probing failed

def probe_video(video_path: str, count_frames: bool = False) -> VideoMetadata:
    """Read header metadata via imageio's ffmpeg backend.

    `count_frames=False` (default) computes frame count as round(duration * fps),
    which is fast and accurate for constant-frame-rate videos. Set True to force
    a full decode pass (slow but exact, useful for variable-frame-rate sources).
    """
    try:
        reader = imageio.get_reader(video_path, "ffmpeg")
        meta = reader.get_meta_data()
        width, height = meta["size"]  # imageio reports (W, H)
        fps = float(meta["fps"])
        duration = float(meta["duration"])
        if count_frames:
            num_frames = reader.count_frames()
        else:
            num_frames = int(round(duration * fps))
        codec = meta.get("codec")
        pix_fmt = meta.get("pix_fmt")
        reader.close()
        return VideoMetadata(
            path=video_path,
            width=width,
            height=height,
            fps=fps,
            duration_sec=duration,
            num_frames=num_frames,
            codec=codec,
            pix_fmt=pix_fmt,
            file_size_mb=osp.getsize(video_path) / (1024 ** 2),
        )
    except Exception as e:
        return VideoMetadata(
            path=video_path,
            width=0, height=0, fps=0.0, duration_sec=0.0, num_frames=0,
            error=f"{type(e).__name__}: {e}",
        )

def build_ffmpeg_cmd(
    input_path: str,
    output_path: str,
    target_fps: int,
    width: Optional[int] = None,
    height: Optional[int] = None,
    crf: int = 18,
    preset: str = "medium",
    threads: int = 0,
) -> List[str]:
    """Compose an ffmpeg command line for FPS resampling.

    Resampling uses the `fps` *filter* (`fps=<n>`), NOT the bare `-r` output
    flag. The `fps` filter rebuilds a constant-rate timeline by dropping/
    duplicating frames against their PTS, so it produces correct-duration CFR
    output even for variable-frame-rate or bad-timestamp sources. The `-r`
    output option, by contrast, leans on the source timebase/PTS and can leave
    VFR sources with their original frame spacing — inflating the muxed
    duration (e.g. a 30s clip muxed as 1440s) while the declared fps still
    reads correctly. `-vsync cfr` forces constant-rate muxing as a backstop
    (the older spelling of `-fps_mode cfr`; works on the older ffmpeg builds
    that imageio_ffmpeg tends to bundle).
    `-crf 18` is visually-lossless x264; lower = higher quality (larger files).

    `threads` caps x264's internal thread count. When running many encodes in
    parallel, set this low (e.g. 2-4) so jobs don't oversubscribe the CPU; x264
    scales poorly past ~8 threads anyway, so per-file threading buys little once
    the machine is already saturated by concurrent jobs. `0` lets ffmpeg decide.
    """
    # Build a single filtergraph: optional scale, then the fps resampler.
    filters = []
    if width is not None and height is not None:
        filters.append(f"scale={width}:{height}")
    filters.append(f"fps={target_fps}")

    cmd = [get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", input_path]
    cmd += [
        "-vf", ",".join(filters),
        "-vsync", "cfr",  # force constant-frame-rate muxing (works on old + new ffmpeg)
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-pix_fmt", "yuv420p",
        "-threads", str(threads),
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
    preset: str = "medium",
    threads: int = 0,
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

    cmd = build_ffmpeg_cmd(input_path, output_path, target_fps, width, height,
                           crf, preset, threads)
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


def _resample_job(job: dict) -> dict:
    """Process pool worker: resample one video and build its curated record.

    Top-level (picklable) so it can run in a ProcessPoolExecutor. Returns a dict
    with the status and curated record on success, or the error on failure —
    exceptions are captured rather than raised so one bad file can't kill the pool.
    """
    raw = job["raw"]
    video = raw.get("video_path") or raw.get("path", "")
    try:
        status = resample_one(
            input_path=video,
            output_path=job["out_path"],
            target_fps=job["target_fps"],
            width=job["width"],
            height=job["height"],
            crf=job["crf"],
            preset=job["preset"],
            threads=job["threads"],
            skip_if_exists=job["skip_if_exists"],
            copy_if_already_match=job["copy_if_already_match"],
        )
        record = make_curated_record(
            raw, job["out_path"], job["target_fps"], job["width"], job["height"]
        )
        return {"video": video, "status": status, "record": record}
    except Exception as e:
        return {"video": video, "status": "failed", "record": None, "error": str(e)}


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
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of videos to encode concurrently. 0 (default) = auto "
                             "(cpu_count // --threads). Set to 1 for the old serial behavior.")
    parser.add_argument("--threads", type=int, default=4,
                        help="x264 threads per encode. Keep small when running many workers; "
                             "workers x threads should roughly match the core count.")
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

    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 1) // max(1, args.threads))
    workers = min(workers, len(records))

    print(f"Resampling {len(records)} video(s) -> {args.target_fps} fps")
    print(f"  Metadata: {metadata_path}")
    print(f"  Output:   {output_dir}")
    print(f"  Workers:  {workers} x {args.threads} thread(s) (preset={args.preset})")
    if args.width is not None:
        print(f"  Resizing to {args.width}x{args.height}")

    counts = {
        "resampled": 0,
        "skipped (exists)": 0,
        "copied (fps already matches)": 0,
        "failed": 0,
    }
    jobs = [
        {
            "raw": raw,
            "out_path": osp.join(
                output_dir,
                relative_output(raw.get("video_path") or raw.get("path", ""), input_dir),
            ),
            "target_fps": args.target_fps,
            "width": args.width,
            "height": args.height,
            "crf": args.crf,
            "preset": args.preset,
            "threads": args.threads,
            "skip_if_exists": not args.overwrite,
            "copy_if_already_match": args.copy_matching,
        }
        for raw in records
    ]

    curated_records: List[dict] = []

    def _tally(result: dict) -> None:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
        if result["record"] is not None:
            curated_records.append(result["record"])
        else:
            print(f"FAILED {result['video']}: {result.get('error', 'unknown error')}")

    if workers == 1:
        for job in tqdm(jobs):
            _tally(_resample_job(job))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_resample_job, job) for job in jobs]
            for fut in tqdm(as_completed(futures), total=len(futures)):
                _tally(fut.result())

    with open(curated_metadata_path, "w", encoding="utf-8") as fh:
        for record in curated_records:
            fh.write(json.dumps(record) + "\n")

    print("\nResult:")
    for status, count in counts.items():
        print(f"  {count:>5}  {status}")
    print(f"\nCurated metadata -> {curated_metadata_path} ({len(curated_records)} records)")


if __name__ == "__main__":
    main()
