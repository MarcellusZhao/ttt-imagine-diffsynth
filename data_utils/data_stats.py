import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def compute_stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def load_metadata(metadata_file: Path) -> list[dict]:
    records = []
    with open(metadata_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def dataset_stats(records: list[dict]) -> dict:
    valid = [r for r in records if "error" not in r and r.get("width") is not None]
    errors = len(records) - len(valid)

    durations = [r["duration"] for r in valid if r.get("duration") is not None]
    fps_vals = [r["fps"] for r in valid if r.get("fps") is not None]
    short_sides = [min(r["width"], r["height"]) for r in valid if r.get("width") and r.get("height")]

    codec_counts = Counter(r.get("codec") for r in valid)
    pix_fmt_counts = Counter(r.get("pix_fmt") for r in valid)

    return {
        "total": len(records),
        "valid": len(valid),
        "errors": errors,
        "duration": compute_stats(durations),
        "fps": compute_stats(fps_vals),
        "short_side": compute_stats(short_sides),
        "codecs": dict(codec_counts.most_common(5)),
        "pix_fmts": dict(pix_fmt_counts.most_common(3)),
    }


def fmt_stat(s: dict) -> str:
    if s["count"] == 0:
        return "N/A"
    return f"mean={s['mean']}, median={s['median']}, min={s['min']}, max={s['max']}"


def print_report(results: dict[str, dict], all_stats: dict | None = None):
    separator = "-" * 72

    for dataset_name, stats in results.items():
        print(separator)
        print(f"Dataset: {dataset_name}")
        print(f"  Total records : {stats['total']}  (valid: {stats['valid']}, errors: {stats['errors']})")
        print(f"  Duration (s)  : {fmt_stat(stats['duration'])}")
        print(f"  FPS           : {fmt_stat(stats['fps'])}")
        print(f"  Short side (px): {fmt_stat(stats['short_side'])}")
        codec_str = ", ".join(f"{k}:{v}" for k, v in stats["codecs"].items())
        print(f"  Codecs        : {codec_str or 'N/A'}")
        pix_str = ", ".join(f"{k}:{v}" for k, v in stats["pix_fmts"].items())
        print(f"  Pix fmts      : {pix_str or 'N/A'}")

    print(separator)

    if all_stats is not None and len(results) > 1:
        print(f"ALL DATASETS ({len(results)} datasets)")
        print(f"  Total records : {all_stats['total']}  (valid: {all_stats['valid']}, errors: {all_stats['errors']})")
        print(f"  Duration (s)  : {fmt_stat(all_stats['duration'])}")
        print(f"  FPS           : {fmt_stat(all_stats['fps'])}")
        print(f"  Short side (px): {fmt_stat(all_stats['short_side'])}")
        print(separator)


def main():
    parser = argparse.ArgumentParser(
        description="Compute video dataset statistics from metadata.jsonl files."
    )
    parser.add_argument(
        "--root",
        help="Root directory containing per-dataset subdirectories with metadata.jsonl, "
             "or a single metadata.jsonl file.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        metavar="NAME",
        help="Only process these dataset subdirectory names (default: all).",
    )
    args = parser.parse_args()

    root = Path(args.root)

    if root.is_file() and root.suffix == ".jsonl":
        metadata_files = [(root.stem, root)]
    elif root.is_dir():
        candidates = sorted(root.glob("*/metadata.jsonl"))
        if not candidates:
            # Maybe root itself has a metadata.jsonl
            direct = root / "metadata.jsonl"
            if direct.exists():
                candidates = [(root.name, direct)]
            else:
                print(f"No metadata.jsonl files found under {root}")
                return
        else:
            candidates = [(p.parent.name, p) for p in candidates]
        if args.datasets:
            keep = set(args.datasets)
            candidates = [(name, p) for name, p in candidates if name in keep]
            if not candidates:
                print(f"None of the requested datasets found under {root}")
                return
        metadata_files = candidates
    else:
        print(f"Path not found or not a directory/jsonl: {root}")
        return

    results = {}
    all_records = []
    for name, path in metadata_files:
        records = load_metadata(path)
        results[name] = dataset_stats(records)
        all_records.extend(records)
        print(f"Loaded {len(records)} records from {path}")

    all_stats = dataset_stats(all_records) if len(results) > 1 else None
    print()
    print_report(results, all_stats)


if __name__ == "__main__":
    main()
