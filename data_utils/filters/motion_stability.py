"""MotionStabilityFilter — reject videos with jittery or shaky-camera footage."""

from typing import Optional

import cv2
import numpy as np

from .base import FilterResult, VideoFilter
from .utils import sample_frames

_EPS = 1e-8


class MotionStabilityFilter(VideoFilter):
    """Reject videos with jittery movement or shaky camera footage.

    Uses Farneback dense optical flow between consecutive sampled frames:
      - jitter_score: coefficient of variation (std/mean) of per-pair flow magnitudes.
        High → erratic, inconsistent motion between frames.
      - flow_coherence: mean alignment of flow unit vectors (|mean(û)|, 0–1).
        Low → flow points in random directions, the signature of camera shake.

    A video passes when jitter_score <= max_jitter_score AND
    flow_coherence >= min_flow_coherence.

    Note: motion analysis benefits from dense temporal sampling.  When used
    inside the curate_videos.py pipeline, pass --n-frames 32 (or higher) for
    reliable statistics; the default pipeline value of 8 gives coarse estimates.
    """

    name = "motion_stability"

    def __init__(
        self,
        max_jitter_score: float = 1.0,
        min_flow_coherence: float = 0.3,
        n_frames: int = 32,
    ):
        self.max_jitter_score = max_jitter_score
        self.min_flow_coherence = min_flow_coherence
        self.n_frames = n_frames

    def _analyse(self, frames: np.ndarray) -> tuple[float, float]:
        """Return (jitter_score, mean_flow_coherence) from sampled frames."""
        grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
        magnitudes: list[float] = []
        coherences: list[float] = []

        for prev, nxt in zip(grays, grays[1:]):
            flow = cv2.calcOpticalFlowFarneback(
                prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            dx, dy = flow[..., 0], flow[..., 1]
            mag = np.sqrt(dx**2 + dy**2)
            magnitudes.append(float(mag.mean()))

            norm = np.maximum(mag, _EPS)
            mask = mag > 0.5  # ignore near-static pixels
            if mask.sum() > 100:
                coherence = float(
                    np.sqrt(
                        (dx[mask] / norm[mask]).mean() ** 2
                        + (dy[mask] / norm[mask]).mean() ** 2
                    )
                )
            else:
                coherence = 1.0  # near-static pair: treat as perfectly coherent
            coherences.append(coherence)

        mag_arr = np.array(magnitudes, dtype=np.float32)
        mean_mag = float(mag_arr.mean())
        jitter_score = float(mag_arr.std() / (mean_mag + _EPS))
        flow_coherence = float(np.mean(coherences))
        return round(jitter_score, 4), round(flow_coherence, 4)

    def __call__(self, video_path: str, frames: Optional[np.ndarray] = None) -> FilterResult:
        if frames is None:
            frames = sample_frames(video_path, self.n_frames)
        if frames is None or len(frames) < 2:
            return FilterResult(passed=False, metadata={"error": "need ≥ 2 frames for motion analysis"})

        jitter_score, flow_coherence = self._analyse(frames)

        passed = (
            jitter_score <= self.max_jitter_score
            and flow_coherence >= self.min_flow_coherence
        )

        return FilterResult(
            passed=passed,
            scores={
                "jitter_score": jitter_score,
                "flow_coherence": flow_coherence,
            },
        )
