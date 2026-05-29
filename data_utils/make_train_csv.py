"""Create a training CSV from captions.jsonl for DatasetFromCSV.

Reads captions.jsonl (fields: video_path, caption) and writes a CSV that
DatasetFromCSV can consume directly. Validates that every video_path exists
before writing. Skips missing files with a warning unless --strict is set.

Usage
-----
Default (bdd100k):
    python data_utils/make_train_csv.py

Custom input/output:
    python data_utils/make_train_csv.py \\
        --input  /path/to/captions.jsonl \\
        --output /path/to/train.csv

Strict mode (fail on any missing video):
    python data_utils/make_train_csv.py --strict
"""

import argparse
import csv
import json
from pathlib import Path


_DEFAULT_INPUT = "/work/nlp/hzhao/datasets/e2e-ttt-video/bdd100k/captions.jsonl"
_DEFAULT_OUTPUT = "/work/nlp/hzhao/datasets/e2e-ttt-video/bdd100k/train.csv"


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
        caption = rec.get("caption", "")
        if not Path(video_path).exists():
            missing.append(video_path)
            if args.strict:
                raise FileNotFoundError(f"Video not found: {video_path}")
            print(f"  WARNING: skipping missing video: {video_path}")
            continue
        rows.append({"video_path": video_path, "caption": caption})

    if missing:
        print(f"\nSkipped {len(missing)} missing video(s).")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["video_path", "caption"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
