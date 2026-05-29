from typing import Optional

import cv2
import numpy as np

from .base import FilterResult, VideoFilter
from .utils import sample_frames


class BlackBorderFilter(VideoFilter):
    """Detect black letterbox/pillarbox borders and reject the video.

    The filter passes when no borders are present, and fails when borders are detected.
    """

    name = "black_border"

    def __init__(
        self,
        brightness_threshold: int = 15,
        min_border_fraction: float = 0.02,
        n_frames: int = 4,
    ):
        self.brightness_threshold = brightness_threshold
        self.min_border_fraction = min_border_fraction
        self.n_frames = n_frames

    def _measure_borders(self, frame: np.ndarray) -> tuple[int, int, int, int]:
        """Return (top, bottom, left, right) black border widths in pixels."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
        h, w = gray.shape
        thresh = self.brightness_threshold

        def scan_rows_forward(limit: int) -> int:
            for i in range(limit):
                if gray[i].mean() > thresh:
                    return i
            return limit

        def scan_rows_backward(limit: int) -> int:
            for i in range(h - 1, h - 1 - limit, -1):
                if gray[i].mean() > thresh:
                    return h - 1 - i
            return limit

        def scan_cols_forward(limit: int) -> int:
            for j in range(limit):
                if gray[:, j].mean() > thresh:
                    return j
            return limit

        def scan_cols_backward(limit: int) -> int:
            for j in range(w - 1, w - 1 - limit, -1):
                if gray[:, j].mean() > thresh:
                    return w - 1 - j
            return limit

        half_h, half_w = h // 2, w // 2
        return (
            scan_rows_forward(half_h),
            scan_rows_backward(half_h),
            scan_cols_forward(half_w),
            scan_cols_backward(half_w),
        )

    def __call__(self, video_path: str, frames: Optional[np.ndarray] = None) -> FilterResult:
        if frames is None:
            frames = sample_frames(video_path, self.n_frames)
        if frames is None or len(frames) == 0:
            return FilterResult(passed=False, metadata={"error": "could not read frames"})

        all_borders = [self._measure_borders(f) for f in frames]
        # Median across frames to suppress single-frame anomalies
        top, bottom, left, right = [int(np.median([b[i] for b in all_borders])) for i in range(4)]

        h, w = frames[0].shape[:2]
        min_px = max(1, int(min(h, w) * self.min_border_fraction))
        has_border = any(b >= min_px for b in [top, bottom, left, right])

        scores = {"border_top": top, "border_bottom": bottom, "border_left": left, "border_right": right}
        return FilterResult(passed=not has_border, scores=scores)
