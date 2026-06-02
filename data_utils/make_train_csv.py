"""Create a DiffSynth-compatible training CSV from captions.jsonl.

Reads captions.jsonl (fields: video_path, caption) and writes a CSV that
DiffSynth-Studio's ``UnifiedDataset`` can consume directly. The output columns
are ``video`` and ``prompt`` — the names DiffSynth expects:

  * ``video`` is the default file-path column declared by ``--data_file_keys``
    (default ``"image,video"``), routed through ``default_video_operator``.
  * ``prompt`` is read verbatim by ``WanTrainingModule.get_pipeline_inputs``
    (see examples/wanvideo/model_training/train.py).

Video paths are written as-is. ``ToAbsolutePath`` joins each path with
``--dataset_base_path`` at load time, but an absolute path is returned
unchanged by ``os.path.join``, so absolute paths in captions.jsonl work
regardless of the base path. Every video_path is validated before writing;
missing files are skipped with a warning unless --strict is set.

Usage
-----
Default (bdd100k):
    python data_utils/make_train_csv.py

Custom input/output:
    python data_utils/make_train_csv.py \\
        --input  /path/to/captions.jsonl \\
        --output /path/to/train_diffsynth.csv

Strict mode (fail on any missing video):
    python data_utils/make_train_csv.py --strict
"""

import argparse
import csv
import json
from pathlib import Path


_DEFAULT_INPUT = "/work/nlp/hzhao/datasets/e2e-ttt-video/bdd100k/captions.jsonl"
_DEFAULT_OUTPUT = "/work/nlp/hzhao/datasets/e2e-ttt-video/bdd100k/train_diffsynth.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        default=_DEFAULT_INPUT,
        help=f"Input captions JSONL (default: {_DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", "-o",
        default=_DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort if any video_path does not exist on disk",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open() as fh:
        records = [json.loads(line) for line in fh if line.strip()]

    print(f"Loaded {len(records)} records from {input_path}")

    rows = []
    missing = []
    for rec in records:
        video_path = rec.get("video_path", "")
        # Captions in the JSONL often carry a leading space; strip it so the
        # prompt fed to the text encoder is clean.
        prompt = rec.get("caption", "").strip()
        if not Path(video_path).exists():
            missing.append(video_path)
            if args.strict:
                raise FileNotFoundError(f"Video not found: {video_path}")
            print(f"  WARNING: skipping missing video: {video_path}")
            continue
        rows.append({"video": video_path, "prompt": prompt})

    if missing:
        print(f"\nSkipped {len(missing)} missing video(s).")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["video", "prompt"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
