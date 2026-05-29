from typing import Optional

import numpy as np

from .base import FilterResult, VideoFilter
from .utils import sample_frames


class TextCoverageFilter(VideoFilter):
    """Estimate the fraction of each frame covered by detected text via EasyOCR.

    Requires: pip install easyocr
    """

    name = "text"

    def __init__(
        self,
        max_coverage: float = 0.1,
        n_frames: int = 4,
        device: str = "cpu",
        languages: list[str] | None = None,
    ):
        self.max_coverage = max_coverage
        self.n_frames = n_frames
        self.device = device
        self.languages = languages or ["en"]
        self._reader = None

    @property
    def reader(self):
        if self._reader is None:
            try:
                import easyocr
            except ImportError:
                raise ImportError(
                    "easyocr is required for TextCoverageFilter. "
                    "Install with: pip install easyocr"
                )
            self._reader = easyocr.Reader(
                self.languages,
                gpu=(self.device != "cpu"),
                verbose=False,
            )
        return self._reader

    def _frame_coverage(self, frame: np.ndarray) -> float:
        h, w = frame.shape[:2]
        total = h * w
        result = self.reader.detect(frame, min_size=10)
        if not result or not result[0]:
            return 0.0
        text_area = 0
        for bbox in result[0]:
            if not bbox:
                continue
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            if not xs or not ys:
                continue
            text_area += max(0, max(xs) - min(xs)) * max(0, max(ys) - min(ys))
        return min(1.0, text_area / total)

    def __call__(self, video_path: str, frames: Optional[np.ndarray] = None) -> FilterResult:
        if frames is None:
            frames = sample_frames(video_path, self.n_frames)
        if frames is None or len(frames) == 0:
            return FilterResult(passed=False, metadata={"error": "could not read frames"})

        coverages = [self._frame_coverage(f) for f in frames]
        avg = float(np.mean(coverages))

        return FilterResult(
            passed=avg <= self.max_coverage,
            scores={"coverage": round(avg, 4)},
        )
