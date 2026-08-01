"""Build a model-comparison contact sheet: rows = models, columns = sampled frames.

Each input video becomes one row; frames are sampled at fixed frame indices and
laid out as columns, with a row label overlaid in the upper left of the first
frame and a timestamp header on top. This reproduces the qualitative comparison grids common in video-generation
papers (one method per row, time progressing left-to-right).

Column headers are labeled in seconds, so `--fps` is required: it converts the
sampled frame indices into timestamps (`frame_index / fps`).

Single comparison (explicit paths + labels, sample every 30 frames, 7 columns):
    python -m data_utils.plot_comparison_grid \\
        --videos pose.mp4 one_to_all.mp4 steadydancer.mp4 ours.mp4 \\
        --labels Pose One-to-All SteadyDancer Ours \\
        --start 0 --step 30 --count 7 --fps 16 \\
        --output comparison.pdf

Explicit frame indices instead of start/step/count:
    python -m data_utils.plot_comparison_grid \\
        --videos a.mp4 b.mp4 --labels A B \\
        --indices 0 48 96 144 192 --fps 16 \\
        --output comparison.pdf

Notes:
- Frames are read by sequential decode (exact), not by codec seeking
  (approximate for inter-frame-coded H.264/H.265). See `sample_frames`.
- If a requested index exceeds a video's length, that cell is left blank and a
  warning is logged; the grid still renders.
"""

import argparse
import logging
import os
from dataclasses import dataclass
from os import path as osp
from typing import Dict, List, Optional

import imageio
import numpy as np
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never open a window
import matplotlib.pyplot as plt

logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class VideoRow:
    """One row of the grid: a label plus the frames sampled for it.

    `frames[i]` is `None` when the requested index ran past the end of the
    video, so downstream rendering can draw a blank placeholder instead.
    """

    label: str
    path: str
    frames: List[Optional[np.ndarray]]


def resolve_indices(args: argparse.Namespace) -> List[int]:
    """Turn CLI args into the sorted, de-duplicated list of frame indices.

    `--indices` (explicit) wins; otherwise build an arithmetic progression from
    `--start`/`--step`/`--count`. We sort+dedup so the sequential reader can
    walk forward once and so two columns never request the same frame twice.
    """
    if args.indices is not None:
        indices = args.indices
    else:
        indices = [args.start + step_i * args.step for step_i in range(args.count)]
    indices = sorted(set(indices))
    if not indices:
        raise ValueError("No frame indices to sample (check --indices/--count).")
    if indices[0] < 0:
        raise ValueError(f"Frame indices must be non-negative, got {indices[0]}.")
    return indices


def sample_frames(video_path: str, indices: List[int]) -> Dict[int, np.ndarray]:
    """Return {index: RGB uint8 frame} for the requested indices.

    We decode sequentially and capture frames as their index matches, breaking
    once we pass the largest requested index. This is exact (unlike seeking,
    which on inter-frame-coded streams can only land on keyframes) and only
    decodes up to the last needed frame rather than the whole file.
    """
    wanted = set(indices)
    last_needed = max(indices)
    captured: Dict[int, np.ndarray] = {}

    reader = imageio.get_reader(video_path, "ffmpeg")
    try:
        for frame_index, frame in enumerate(reader):
            if frame_index in wanted:
                captured[frame_index] = np.asarray(frame)
            if frame_index >= last_needed:
                break
    finally:
        reader.close()

    missing = sorted(wanted - captured.keys())
    if missing:
        logger.warning(
            "%s: frame indices %s exceed the video length (got %d frames); "
            "those cells will be blank.",
            osp.basename(video_path),
            missing,
            max(captured.keys()) + 1 if captured else 0,
        )
    return captured


def build_rows(
    video_paths: List[str], labels: List[str], indices: List[int]
) -> List[VideoRow]:
    rows: List[VideoRow] = []
    for video_path, label in tqdm(
        list(zip(video_paths, labels)), desc="Sampling videos"
    ):
        if not osp.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        captured = sample_frames(video_path, indices)
        frames = [captured.get(frame_index) for frame_index in indices]
        rows.append(VideoRow(label=label, path=video_path, frames=frames))
    return rows


def column_header(frame_index: int, fps: float) -> str:
    """Label a column by its timestamp in seconds rather than its frame index."""
    return f"{frame_index / fps:.1f}s"


def render_grid(
    rows: List[VideoRow],
    indices: List[int],
    output_path: str,
    fps: float,
    cell_width_in: float,
    dpi: int,
) -> None:
    """Lay the sampled frames out as a matplotlib subplot grid and save to file.

    matplotlib (rather than hand-pasting into one big array) gives us free
    label overlays, column titles, and uniform spacing. Figure height is derived from
    the first real frame's aspect ratio so cells aren't stretched.
    """
    num_rows = len(rows)
    num_cols = len(indices)

    aspect = 9 / 16  # height/width fallback until we see a real frame
    for row in rows:
        for frame in row.frames:
            if frame is not None:
                aspect = frame.shape[0] / frame.shape[1]
                break
        else:
            continue
        break

    fig_width = cell_width_in * num_cols
    fig_height = cell_width_in * aspect * num_rows
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,  # always index axes[r][c], even for a 1xN or Nx1 grid
    )

    for row_index, row in enumerate(rows):
        for col_index, frame in enumerate(row.frames):
            ax = axes[row_index][col_index]
            if frame is not None:
                ax.imshow(frame)
            else:
                # Blank placeholder keeps the grid aligned for missing frames.
                ax.imshow(np.ones((10, 10, 3), dtype=np.uint8) * 255)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if row_index == 0:
                ax.set_title(column_header(indices[col_index], fps), fontsize=8)
            if col_index == 0:
                # Label sits inside the first frame's upper-left corner; the
                # translucent box keeps it readable over busy video content.
                ax.text(
                    0.03,
                    0.95,
                    row.label,
                    transform=ax.transAxes,
                    fontsize=6,
                    ha="left",
                    va="top",
                    color="white",
                    bbox=dict(
                        facecolor="black", alpha=0.5, pad=2, edgecolor="none"
                    ),
                )

    fig.subplots_adjust(wspace=0.02, hspace=0.02)
    os.makedirs(osp.dirname(osp.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    logger.info("Wrote %s (%d rows x %d cols)", output_path, num_rows, num_cols)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a model-comparison frame grid from several videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--videos", nargs="+", required=True, help="Video paths, one per row."
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Row labels (default: video filename stems).",
    )

    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        default=None,
        help="Explicit frame indices to sample; overrides --start/--step/--count.",
    )
    parser.add_argument("--start", type=int, default=0, help="First frame index.")
    parser.add_argument("--step", type=int, default=30, help="Frame index stride.")
    parser.add_argument("--count", type=int, default=7, help="Number of columns.")

    parser.add_argument(
        "--fps",
        type=float,
        required=True,
        help="Frame rate of the videos; converts frame indices to the "
        "timestamps shown as column headers (frame_index / fps).",
    )
    parser.add_argument(
        "--cell-width-in",
        type=float,
        default=2.0,
        help="Per-cell width in inches (height follows the frame aspect ratio).",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Output resolution.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output path; saved as PDF (a non-.pdf extension is replaced).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.fps <= 0:
        raise ValueError(f"--fps must be positive, got {args.fps}.")

    if args.labels is None:
        labels = [osp.splitext(osp.basename(path))[0] for path in args.videos]
    else:
        if len(args.labels) != len(args.videos):
            raise ValueError(
                f"--labels has {len(args.labels)} entries but --videos has "
                f"{len(args.videos)}; they must match."
            )
        labels = args.labels

    # PDF keeps the labels/headers as vector text, so they stay sharp at any
    # zoom level; rewrite whatever extension was passed.
    output_root, output_ext = osp.splitext(args.output)
    output_path = output_root + ".pdf"
    if output_ext.lower() != ".pdf":
        logger.info("Saving as PDF: %s -> %s", args.output, output_path)

    indices = resolve_indices(args)
    logger.info("Sampling frame indices: %s", indices)

    rows = build_rows(args.videos, labels, indices)
    render_grid(
        rows,
        indices,
        output_path=output_path,
        fps=args.fps,
        cell_width_in=args.cell_width_in,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
