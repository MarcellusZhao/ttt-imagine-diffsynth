"""Copy curated videos to a destination directory and update curated.jsonl.

For each entry in curated.jsonl:
  - Copies the video to <dest-path>/raw_videos/<filename>
  - Writes an updated raw_metadata.jsonl to <dest-path>/raw_metadata.jsonl with
    "video_path" values rewritten to the new absolute locations.

Usage
-----
Copy all curated videos (default dest):
    python data_utils/copy_curated_videos.py --input curated.jsonl \\
        --dest-path /work/nlp/hzhao/datasets/e2e-ttt-video/<dataset-name>

Dry run (print what would be copied without doing it):
    python data_utils/copy_curated_videos.py --input curated.jsonl \\
        --dest-path /work/nlp/hzhao/datasets/e2e-ttt-video/<dataset-name> --dry-run

Parallel copy with 8 workers:
    python data_utils/copy_curated_videos.py --input curated.jsonl \\
        --dest-path /work/nlp/hzhao/datasets/e2e-ttt-video/<dataset-name> --workers 8
"""

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


def _copy_one(idx: int, video_path: str, dest: Path, dry_run: bool) -> tuple[int, str, bool, str]:
    """Copy a single video. Returns (idx, video_path, success, message)."""
    src = Path(video_path)
    if not src.exists():
        return idx, video_path, False, "source not found"
    if dry_run:
        return idx, video_path, True, f"→ {dest}"
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return idx, video_path, True, f"→ {dest}"
    except Exception as exc:
        return idx, video_path, False, f"{type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, help="Input curated.jsonl")
    parser.add_argument(
        "--dest-path", required=True,
        help="Destination dataset directory (e.g. /work/.../e2e-ttt-video/<dataset-name>)",
    )
    parser.add_argument(
        "--workers", "-j", type=int, default=4,
        help="Parallel copy threads (default: 4)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be copied without actually copying",
    )
    args = parser.parse_args()

    dest_path = Path(args.dest_path).resolve()
    raw_videos_dir = dest_path / "raw_videos"
    output_jsonl = dest_path / "raw_metadata.jsonl"

    with open(args.input) as fh:
        records = [json.loads(line) for line in fh if line.strip()]

    print(f"Records   : {len(records)}")
    print(f"Videos → : {raw_videos_dir}")
    print(f"JSONL  → : {output_jsonl}{' (will replace)' if output_jsonl.exists() else ''}")
    if args.dry_run:
        print("Mode      : DRY RUN — no files will be copied\n")

    # Build tasks: (idx, src_path, dest_path)
    tasks: list[tuple[int, str, Path]] = []
    for idx, rec in enumerate(records):
        vpath = rec.get("video_path") or rec.get("path", "")
        if not vpath:
            continue
        dest = raw_videos_dir / Path(vpath).name
        tasks.append((idx, vpath, dest))

    # updated_paths[idx] = new absolute path (only set on success)
    updated_paths: dict[int, str] = {}
    errors: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_copy_one, idx, vpath, dest, args.dry_run): idx
            for idx, vpath, dest in tasks
        }
        with tqdm(total=len(futures), desc="Copying") as pbar:
            for future in as_completed(futures):
                idx, vpath, success, msg = future.result()
                if success:
                    # Resolve the dest for this task to get the new path
                    dest = raw_videos_dir / Path(vpath).name
                    updated_paths[idx] = str(dest)
                else:
                    errors.append((vpath, msg))
                pbar.update(1)

    ok = len(updated_paths)
    err = len(errors)
    print(f"\nDone: {ok} copied, {err} failed")
    if errors:
        print("\nFailed:")
        for vpath, msg in errors:
            print(f"  {vpath}: {msg}")

    # Write updated curated.jsonl with new video_path values
    if not args.dry_run:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        if output_jsonl.exists():
            print(f"Replacing existing {output_jsonl}")
        with output_jsonl.open("w") as fout:
            for idx, rec in enumerate(records):
                if idx in updated_paths:
                    rec = {**rec, "video_path": updated_paths[idx]}
                fout.write(json.dumps(rec) + "\n")
        print(f"Wrote {output_jsonl}")
    else:
        exists_note = " (would replace existing)" if output_jsonl.exists() else ""
        print(f"\n[dry-run] Would write updated raw_metadata.jsonl to {output_jsonl}{exists_note}")


if __name__ == "__main__":
    main()
