from typing import Optional

import cv2
import numpy as np

from .base import FilterResult, VideoFilter
from .utils import sample_frames


class BlurFilter(VideoFilter):
    """Score sharpness via Laplacian variance; low scores indicate excessive blur."""

    name = "blur"

    def __init__(self, min_score: float = 50.0, n_frames: int = 8):
        self.min_score = min_score
        self.n_frames = n_frames

    def _laplacian_var(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def __call__(self, video_path: str, frames: Optional[np.ndarray] = None) -> FilterResult:
        if frames is None:
            frames = sample_frames(video_path, self.n_frames)
        if frames is None or len(frames) == 0:
            return FilterResult(passed=False, metadata={"error": "could not read frames"})

        scores = [self._laplacian_var(f) for f in frames]
        avg = float(np.mean(scores))

        return FilterResult(
            passed=avg >= self.min_score,
            scores={"score": round(avg, 2)},
        )
