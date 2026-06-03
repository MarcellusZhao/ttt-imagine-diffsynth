"""Video curation pipeline: score and filter videos across multiple quality dimensions.

Input:  JSONL produced by locate_videos.py (one JSON object per line, must have
        a "video_path" field).
Output: JSONL of videos that passed all active filters, with per-filter score/flag
        fields appended to each record.

Filters are applied sequentially in the order below; the survivor list shrinks at
each stage so later (more expensive) filters only run on videos that already passed
the cheaper earlier ones.

Default filter order
--------------------
  1. duration_resolution  — duration ≥ 10 s, aspect ratio ≤ 2.5:1
  2. black_border         — detect & crop black borders in-place (skipped in --dry-run)
  3. overexposure         — over-/under-exposed pixel fraction ≤ 5 %
  4. blur                 — Laplacian variance ≥ 50
  5. motion_stability     — jitter score ≤ 1.0, flow coherence ≥ 0.3
  6. text                 — text coverage ≤ 10 % of frame  (requires easyocr)
  7. aesthetic            — LAION aesthetic score ≥ 4.5     (requires open_clip)
  8. nsfw                 — NSFW content score ≤ 0.2        (requires transformers)

Usage
-----
Run all filters (default order):
    python data_utils/curate_videos.py \\
        --input /path/to/metadata.jsonl \\
        --output /path/to/curated.jsonl \\
        --device cuda

Select specific filters (applied in the order listed):
    python data_utils/curate_videos.py -i metadata.jsonl -o out.jsonl \\
        --filters duration_resolution,blur,overexposure,black_border

Override per-filter thresholds with filter_name.param=value:
    python data_utils/curate_videos.py -i metadata.jsonl -o out.jsonl \\
        --override aesthetic.min_score=4.0 nsfw.max_score=0.2 blur.min_score=100

Output fields added per filter (prefix = "filter_<name>_"):
  <prefix>pass         bool  — whether this filter passed
  <prefix><score_key>  float — numeric score(s) for the filter
  <prefix>crop_box     list  — [x1, y1, x2, y2] crop recommendation (black_border)

  filter_pass          bool  — True only if ALL active filters passed
"""

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from filters import FILTER_REGISTRY, VideoFilter
from filters.utils import sample_frames

_ALL_FILTERS = list(FILTER_REGISTRY.keys())

# Ordered from cheapest/most-aggressive to most-expensive so each stage processes
# only the videos that survived all previous stages.
_DEFAULT_FILTER_ORDER = [
    "duration_resolution",
    "black_border",
    "overexposure",
    "blur",
    "motion_stability",
    "text",
    "aesthetic",
    "nsfw",
]

# Default constructor kwargs per filter (matches class __init__ defaults)
_DEFAULTS: dict[str, dict[str, Any]] = {
    "duration_resolution": {"min_duration": 10.0, "max_aspect_ratio": 2.5},
    "text":                {"max_coverage": 0.1, "n_frames": 8},
    "aesthetic":           {"min_score": 4.5, "n_frames": 8},
    # "watermark":           {"max_score": 0.5, "n_frames": 4, "corner_fraction": 0.15},
    "black_border":        {"brightness_threshold": 15, "min_border_fraction": 0.02, "n_frames": 4},
    "overexposure":        {"max_overexposed_fraction": 0.05, "max_underexposed_fraction": 0.05, "n_frames": 8},
    "blur":                {"min_score": 50.0, "n_frames": 8},
    "motion_stability":    {"max_jitter_score": 1.2, "min_flow_coherence": 0.2, "n_frames": 32},
    "nsfw":                {"max_score": 0.2, "n_frames": 8},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_overrides(items: list[str]) -> dict[str, dict[str, Any]]:
    """Parse ['aesthetic.min_score=4.0', 'blur.min_score=100'] → nested dict."""
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        key, sep, val = item.partition("=")
        if not sep:
            print(f"Warning: skipping malformed override '{item}' (expected key=value)")
            continue
        filter_name, _, param = key.partition(".")
        if not param:
            print(f"Warning: skipping override '{item}' (expected filter_name.param=value)")
            continue
        try:
            parsed: Any = int(val) if val.lstrip("-").isdigit() else float(val)
        except ValueError:
            parsed = val
        result.setdefault(filter_name, {})[param] = parsed
    return result


def _build_filters(
    names: list[str],
    device: str,
    overrides: dict[str, dict[str, Any]],
    dry_run: bool = False,
) -> list[VideoFilter]:
    filters = []
    for name in names:
        if name not in FILTER_REGISTRY:
            raise ValueError(f"Unknown filter '{name}'. Available: {_ALL_FILTERS}")
        cls = FILTER_REGISTRY[name]
        kwargs = {**_DEFAULTS.get(name, {}), **overrides.get(name, {})}
        if "device" in inspect.signature(cls.__init__).parameters:
            kwargs["device"] = device
        f = cls(**kwargs)
        f.dry_run = dry_run
        filters.append(f)
    return filters


def _apply_filter(record: dict, f: VideoFilter, n_frames: int) -> bool:
    """Run one filter on one record, updating it in-place. Returns whether it passed."""
    video_path = record.get("video_path") or record.get("path", "")
    shared = sample_frames(video_path, n_frames)
    try:
        result = f(video_path, frames=shared)
    except Exception as exc:
        prefix = f"filter_{f.name}_"
        record[f"{prefix}pass"] = False
        record[f"{prefix}error"] = f"{type(exc).__name__}: {exc}"
        return False
    record.update(result.to_dict(prefix=f"filter_{f.name}_"))
    return result.passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input",   "-i", required=True,  help="Input JSONL file (from locate_videos.py)")
    parser.add_argument("--output",  "-o", required=True,  help="Output JSONL file with surviving records")
    parser.add_argument(
        "--filters", "-f",
        default=",".join(_DEFAULT_FILTER_ORDER),
        help=f"Comma-separated filters to run in order. Default: {_DEFAULT_FILTER_ORDER}",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Compute device for model-based filters: 'cpu', 'cuda', 'cuda:1', etc.",
    )
    parser.add_argument(
        "--n-frames", type=int, default=8,
        help="Frames to pre-sample per video for each filter stage (default: 8)",
    )
    parser.add_argument(
        "--override", "-O", nargs="*", default=[], metavar="filter.param=value",
        help="Per-filter threshold overrides, e.g. --override aesthetic.min_score=4.0",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Score and report without modifying any files (black_border crop is skipped)",
    )
    args = parser.parse_args()

    filter_names = [n.strip() for n in args.filters.split(",") if n.strip()]
    overrides = _parse_overrides(args.override or [])
    filters = _build_filters(filter_names, args.device, overrides, dry_run=args.dry_run)

    print(f"Filters : {[f.name for f in filters]}")
    print(f"Device  : {args.device}")
    print(f"Frames  : {args.n_frames} per video per stage")
    if args.dry_run:
        print("Mode    : DRY RUN — no files will be modified")

    # Load records
    with open(args.input) as fh:
        all_records = [json.loads(line) for line in fh if line.strip()]
    print(f"Videos  : {len(all_records)} total")

    # Drop records with inaccessible paths before any filtering
    survivors: list[dict] = []
    for record in all_records:
        vpath = record.get("video_path") or record.get("path", "")
        if vpath and Path(vpath).exists():
            survivors.append(record)
        else:
            record["filter_pass"] = False
            record["filter_error"] = "video path missing or inaccessible"
    print(f"          {len(survivors)} accessible\n")

    # Apply filters sequentially — shrink the list at each stage
    stage_stats: list[tuple[str, int, int]] = []
    for f in filters:
        n_before = len(survivors)
        next_survivors: list[dict] = []
        for record in tqdm(survivors, desc=f"Filtering by {f.name:<25}"):
            if _apply_filter(record, f, args.n_frames):
                next_survivors.append(record)
        survivors = next_survivors
        stage_stats.append((f.name, n_before, len(survivors)))

    # Mark all survivors as having passed every active filter
    for record in survivors:
        record["filter_pass"] = True

    # Write survivors to output
    output_path = Path(args.output)
    with output_path.open("w") as fout:
        for record in survivors:
            fout.write(json.dumps(record) + "\n")

    # Summary funnel
    total = len(all_records)
    print(f"\n{'='*55}")
    print(f"{'Filter':<25}  {'In':>6}  {'Out':>6}  {'Kept':>5}")
    print(f"{'-'*55}")
    for name, n_before, n_after in stage_stats:
        pct = 100 * n_after // max(n_before, 1)
        print(f"  {name:<23}  {n_before:>6}  {n_after:>6}  {pct:>4}%")
    print(f"{'-'*55}")
    pct = 100 * len(survivors) // max(total, 1)
    print(f"  {'OVERALL':<23}  {total:>6}  {len(survivors):>6}  {pct:>4}%")
    if args.dry_run:
        print("(DRY RUN — output scores reflect what would have been filtered/cropped)")
    print(f"{'='*55}")
    print(f"Output → {output_path}")


if __name__ == "__main__":
    main()
