from typing import Optional

import numpy as np

from .base import FilterResult, VideoFilter
from .utils import sample_frames


class OverexposureFilter(VideoFilter):
    """Filter videos with abnormal tonal distributions (blown highlights or crushed blacks)."""

    name = "overexposure"

    def __init__(
        self,
        max_overexposed_fraction: float = 0.05,
        max_underexposed_fraction: float = 0.05,
        overexposed_threshold: int = 250,
        underexposed_threshold: int = 5,
        n_frames: int = 8,
    ):
        self.max_over = max_overexposed_fraction
        self.max_under = max_underexposed_fraction
        self.over_thresh = overexposed_threshold
        self.under_thresh = underexposed_threshold
        self.n_frames = n_frames

    def _analyze(self, frame: np.ndarray) -> tuple[float, float]:
        # Luminance approximation via channel mean
        luma = frame.mean(axis=2)
        total = luma.size
        return (
            float(np.sum(luma > self.over_thresh) / total),
            float(np.sum(luma < self.under_thresh) / total),
        )

    def __call__(self, video_path: str, frames: Optional[np.ndarray] = None) -> FilterResult:
        if frames is None:
            frames = sample_frames(video_path, self.n_frames)
        if frames is None or len(frames) == 0:
            return FilterResult(passed=False, metadata={"error": "could not read frames"})

        pairs = [self._analyze(f) for f in frames]
        avg_over = float(np.mean([p[0] for p in pairs]))
        avg_under = float(np.mean([p[1] for p in pairs]))

        return FilterResult(
            passed=avg_over <= self.max_over and avg_under <= self.max_under,
            scores={
                "overexposed_fraction": round(avg_over, 4),
                "underexposed_fraction": round(avg_under, 4),
            },
        )
