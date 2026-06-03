#!/usr/bin/env python3
"""Central launcher for the per-dataset video data-processing pipeline.

All the ``video_filter_*.sh`` / ``video_caption_*.sh`` scripts under
``data_utils/`` run the same fixed sequence of steps; only the raw source
directory and the destination directory differ per dataset. This script
collapses them into one place so you can pick *which datasets* and *which
steps* to run.

Pipeline (in order):
    1. locate    locate_videos.py      -> <raw>/metadata.jsonl
    2. curate    curate_videos.py      -> <raw>/curated.jsonl   (model-based filtering)
    3. copy      copy_curated_videos.py-> copies survivors into <dest>/
    4. resample  resample_fps.py       -> re-encode <dest> videos to --target-fps
    5. caption   generate_caption.py   -> <dest>/captions.jsonl
    6. csv       make_train_csv.py     -> <dest>/train_diffsynth.csv

Examples
--------
    # Full pipeline on two datasets
    python data_utils/run_data_pipeline.py --datasets panda ego4d

    # Only (re)generate captions + train csv for every dataset
    python data_utils/run_data_pipeline.py --datasets all --steps caption csv

    # See the exact commands without running anything
    python data_utils/run_data_pipeline.py --datasets all --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
DATA_UTILS_DIR = Path(__file__).resolve().parent          # .../data_utils
REPO_ROOT = DATA_UTILS_DIR.parent                          # .../ttt-imagine-diffsynth

# Filenames produced/consumed between steps (kept here so the whole convention
# lives in one place). NOTE: these mirror what the original shell scripts pass;
# in particular `generate_caption.py` is fed CAPTION_INPUT below.
METADATA_NAME = "metadata.jsonl"          # locate -> raw
CURATED_NAME = "curated.jsonl"            # curate -> raw
CAPTION_INPUT = "curated_metadata.jsonl"  # generate_caption reads this from <dest>
CAPTIONS_NAME = "captions.jsonl"          # generate_caption -> <dest>
TRAIN_CSV_NAME = "train_diffsynth.csv"    # make_train_csv -> <dest>


# --------------------------------------------------------------------------- #
# Dataset registry: raw source dir + processed destination dir.
# Add a new dataset by adding one entry here.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Dataset:
    name: str
    raw_root: str   # where the original videos + metadata.jsonl live
    dest_root: str  # where curated/resampled videos + train csv are written


_DATASETS_BASE = "/work/nlp/hzhao/datasets"
_DEST_BASE = "/work/nlp/hzhao/datasets/e2e-ttt-video"

DATASETS: dict[str, Dataset] = {
    d.name: d
    for d in [
        Dataset("bdd100k",          f"{_DATASETS_BASE}/ShareGPT4Video/bdd100k",            f"{_DEST_BASE}/bdd100k"),
        Dataset("panda",            f"{_DATASETS_BASE}/ShareGPT4Video/panda",              f"{_DEST_BASE}/panda"),
        Dataset("pixabay",          f"{_DATASETS_BASE}/pixabay_v2_tar",                    f"{_DEST_BASE}/pixabay"),
        Dataset("mixkit",           f"{_DATASETS_BASE}/all_mixkit",                        f"{_DEST_BASE}/mixkit"),
        Dataset("ego4d",            f"{_DATASETS_BASE}/ego4d",                             f"{_DEST_BASE}/ego4d"),
        Dataset("agibotworld2026",  f"{_DATASETS_BASE}/AgiBotWorld2026/ImitationLearning", f"{_DEST_BASE}/agibotworld2026"),
        Dataset("physicalai-av",    f"{_DATASETS_BASE}/PhysicalAI-Autonomous-Vehicles",    f"{_DEST_BASE}/physicalai-av"),
    ]
}


# --------------------------------------------------------------------------- #
# Step definitions: each builds the argv (minus the leading `python <script>`)
# for one dataset given the parsed CLI args.
# --------------------------------------------------------------------------- #
def _step_locate(ds: Dataset, a: argparse.Namespace) -> list[str]:
    return ["locate_videos.py", "-r", ds.raw_root]


def _step_curate(ds: Dataset, a: argparse.Namespace) -> list[str]:
    return [
        "curate_videos.py",
        "--input", f"{ds.raw_root}/{METADATA_NAME}",
        "--output", f"{ds.raw_root}/{CURATED_NAME}",
        "--device", a.device,
    ]


def _step_copy(ds: Dataset, a: argparse.Namespace) -> list[str]:
    return [
        "copy_curated_videos.py",
        "--input", f"{ds.raw_root}/{CURATED_NAME}",
        "--dest-path", ds.dest_root,
        "--workers", str(a.workers),
    ]


def _step_resample(ds: Dataset, a: argparse.Namespace) -> list[str]:
    return [
        "resample_fps.py",
        "--dataset_path", ds.dest_root,
        "--target_fps", str(a.target_fps),
    ]


def _step_caption(ds: Dataset, a: argparse.Namespace) -> list[str]:
    return [
        "generate_caption.py",
        "--input", f"{ds.dest_root}/{CAPTION_INPUT}",
    ]


def _step_csv(ds: Dataset, a: argparse.Namespace) -> list[str]:
    return [
        "make_train_csv.py",
        "--input", f"{ds.dest_root}/{CAPTIONS_NAME}",
        "--output", f"{ds.dest_root}/{TRAIN_CSV_NAME}",
    ]


# Ordered: this is the canonical pipeline order.
STEPS: dict[str, callable] = {
    "locate": _step_locate,
    "curate": _step_curate,
    "copy": _step_copy,
    "resample": _step_resample,
    "caption": _step_caption,
    "csv": _step_csv,
}
STEP_ORDER = list(STEPS.keys())


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def build_argv(ds: Dataset, step: str, a: argparse.Namespace) -> list[str]:
    script_args = STEPS[step](ds, a)
    script_path = str(DATA_UTILS_DIR / script_args[0])
    return [sys.executable, script_path, *script_args[1:]]


def run(argv: list[str], dry_run: bool) -> int:
    pretty = " ".join(argv)
    print(f"  $ {pretty}", flush=True)
    if dry_run:
        return 0
    return subprocess.run(argv, cwd=REPO_ROOT).returncode


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--datasets", "-d", nargs="+", required=True, metavar="NAME",
        help="Datasets to process. Use 'all' for every registered dataset. "
             f"Available: {', '.join(DATASETS)}",
    )
    p.add_argument(
        "--steps", "-s", nargs="+", default=STEP_ORDER, metavar="STEP",
        help=f"Pipeline steps to run, in pipeline order. Default: all. "
             f"Available: {', '.join(STEP_ORDER)}",
    )
    p.add_argument("--target-fps", type=int, default=16, help="resample target fps (default: 16)")
    p.add_argument("--device", default="cuda", help="curate compute device (default: cuda)")
    p.add_argument("--workers", type=int, default=4, help="copy parallel threads (default: 4)")
    p.add_argument("--dry-run", action="store_true", help="print commands without executing")
    p.add_argument(
        "--continue-on-error", action="store_true",
        help="keep going to the next step/dataset if a command fails",
    )
    a = p.parse_args()

    # Resolve dataset selection.
    if "all" in a.datasets:
        selected = list(DATASETS.values())
    else:
        unknown = [d for d in a.datasets if d not in DATASETS]
        if unknown:
            p.error(f"unknown dataset(s): {', '.join(unknown)}. Available: {', '.join(DATASETS)}")
        selected = [DATASETS[d] for d in a.datasets]

    # Resolve + order step selection (always runs in canonical pipeline order).
    unknown_steps = [s for s in a.steps if s not in STEPS]
    if unknown_steps:
        p.error(f"unknown step(s): {', '.join(unknown_steps)}. Available: {', '.join(STEP_ORDER)}")
    steps = [s for s in STEP_ORDER if s in set(a.steps)]

    print(f"Datasets : {', '.join(d.name for d in selected)}")
    print(f"Steps    : {', '.join(steps)}")
    print(f"Mode     : {'DRY RUN' if a.dry_run else 'EXECUTE'}\n")

    failures: list[str] = []
    for ds in selected:
        print(f"=== {ds.name} ===")
        for step in steps:
            argv = build_argv(ds, step, a)
            t0 = time.time()
            rc = run(argv, a.dry_run)
            if rc != 0:
                tag = f"{ds.name}:{step}"
                failures.append(tag)
                print(f"  ! step '{step}' failed (exit {rc})", flush=True)
                if not a.continue_on_error:
                    print(f"\nAborting at {tag}. Use --continue-on-error to skip failures.")
                    return rc
            elif not a.dry_run:
                print(f"  done '{step}' in {time.time() - t0:.1f}s", flush=True)
        print()

    if failures:
        print(f"Completed with {len(failures)} failure(s): {', '.join(failures)}")
        return 1
    print("All done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
