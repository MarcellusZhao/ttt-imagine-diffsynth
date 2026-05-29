"""Motion quality analysis for video datasets.

Intended as a second-pass step, run on videos that have already passed the
first-stage curation filters.  Scores each video on two axes:

  motion_score  — mean absolute pixel difference between consecutive frames
                  (0–255 scale).  Low → static or near-static video.

  jitter_score  — coefficient of variation (std / mean) of per-frame motion
                  scores.  High → erratic, unstable (jittery) motion.

When use_optical_flow=True, Farneback dense optical flow replaces frame
differencing for motion_score, and adds flow_coherence (alignment of flow
vectors, 0–1; low → incoherent / unnatural motion) as an informational score.

Usage
-----
    python data_utils/motion.py \\
        --input curated.jsonl \\
        --output motion_scored.jsonl \\
        --min-motion-score 2.0 \\
        --max-jitter-score 1.0
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from data_utils.filters.utils import sample_frames

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _frame_diffs(frames: np.ndarray) -> np.ndarray:
    """Mean absolute grayscale difference for each consecutive frame pair, shape (N-1,)."""
    gray = np.stack(
        [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32) for f in frames]
    )
    return np.mean(np.abs(np.diff(gray, axis=0)), axis=(1, 2))


def _optical_flow(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        magnitudes  — mean flow magnitude per frame pair, shape (N-1,)
        coherences  — mean direction alignment per frame pair, shape (N-1,)
                      |mean(unit flow vectors)|; 1 = all pointing the same way.
    """
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    magnitudes: list[float] = []
    coherences: list[float] = []
    for prev, nxt in zip(grays, grays[1:]):
        flow = cv2.calcOpticalFlowFarneback(
            prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        dx, dy = flow[..., 0], flow[..., 1]
        mag = np.sqrt(dx ** 2 + dy ** 2)
        magnitudes.append(float(mag.mean()))

        norm = np.maximum(mag, _EPS)
        mask = mag > 0.5  # ignore near-zero vectors
        if mask.sum() > 100:
            coherence = float(
                np.sqrt((dx[mask] / norm[mask]).mean() ** 2 + (dy[mask] / norm[mask]).mean() ** 2)
            )
        else:
            coherence = 0.0
        coherences.append(coherence)

    return np.array(magnitudes), np.array(coherences)


# ---------------------------------------------------------------------------
# Analyser class
# ---------------------------------------------------------------------------

@dataclass
class MotionResult:
    passed: bool
    motion_score: float
    jitter_score: float
    flow_coherence: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        out = {
            "motion_pass": self.passed,
            "motion_score": self.motion_score,
            "motion_jitter_score": self.jitter_score,
        }
        if self.flow_coherence is not None:
            out["motion_flow_coherence"] = self.flow_coherence
        if self.error is not None:
            out["motion_error"] = self.error
        return out


class MotionAnalyser:
    """Score videos for motion quantity and stability."""

    def __init__(
        self,
        min_motion_score: float = 2.0,
        max_jitter_score: float = 1.0,
        use_optical_flow: bool = False,
        n_frames: int = 32,
    ):
        self.min_motion_score = min_motion_score
        self.max_jitter_score = max_jitter_score
        self.use_optical_flow = use_optical_flow
        self.n_frames = n_frames

    def __call__(self, video_path: str, frames: Optional[np.ndarray] = None) -> MotionResult:
        if frames is None:
            frames = sample_frames(video_path, self.n_frames)
        if frames is None or len(frames) < 2:
            return MotionResult(passed=False, motion_score=0.0, jitter_score=0.0,
                                error="need ≥ 2 frames for motion analysis")

        if self.use_optical_flow:
            motion_scores, coherences = _optical_flow(frames)
            flow_coherence = round(float(np.mean(coherences)), 4)
        else:
            motion_scores = _frame_diffs(frames)
            flow_coherence = None

        motion_score = round(float(np.mean(motion_scores)), 4)
        jitter_score = round(float(np.std(motion_scores) / (motion_score + _EPS)), 4)

        passed = (
            motion_score >= self.min_motion_score
            and jitter_score <= self.max_jitter_score
        )

        return MotionResult(
            passed=passed,
            motion_score=motion_score,
            jitter_score=jitter_score,
            flow_coherence=flow_coherence,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input",  "-i", required=True, help="Input JSONL (output of curate_videos.py)")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL with motion scores appended")
    parser.add_argument("--min-motion-score", type=float, default=2.0,
                        help="Minimum mean frame difference to pass (default: 2.0)")
    parser.add_argument("--max-jitter-score", type=float, default=1.0,
                        help="Maximum motion CV to pass (default: 1.0)")
    parser.add_argument("--optical-flow", action="store_true",
                        help="Use Farneback dense optical flow instead of frame differencing")
    parser.add_argument("--n-frames", type=int, default=32,
                        help="Frames to sample per video (default: 32)")
    parser.add_argument("--filter-passed-only", action="store_true",
                        help="Only process records where filter_pass=True")
    args = parser.parse_args()

    analyser = MotionAnalyser(
        min_motion_score=args.min_motion_score,
        max_jitter_score=args.max_jitter_score,
        use_optical_flow=args.optical_flow,
        n_frames=args.n_frames,
    )

    with open(args.input) as fh:
        records = [json.loads(line) for line in fh if line.strip()]

    if args.filter_passed_only:
        records = [r for r in records if r.get("filter_pass", True)]
        print(f"Processing {len(records)} records with filter_pass=True")
    else:
        print(f"Processing {len(records)} records")

    stats = {"total": 0, "pass": 0}
    with open(args.output, "w") as fout:
        for record in tqdm(records, desc="Motion analysis"):
            video_path = record.get("video_path") or record.get("path", "")
            if not video_path or not Path(video_path).exists():
                record.update({"motion_pass": False, "motion_error": "path missing or inaccessible"})
            else:
                result = analyser(video_path)
                record.update(result.to_dict())

            fout.write(json.dumps(record) + "\n")
            stats["total"] += 1
            if record.get("motion_pass"):
                stats["pass"] += 1

    total = max(stats["total"], 1)
    print(f"\nMotion pass: {stats['pass']}/{stats['total']} ({100*stats['pass']//total}%)")
    print(f"Output → {args.output}")


if __name__ == "__main__":
    main()
