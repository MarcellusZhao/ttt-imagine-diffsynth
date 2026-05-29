from typing import Optional

import logging

import imageio
import numpy as np

logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)

from .base import FilterResult, VideoFilter


class DurationResolutionFilter(VideoFilter):
    """Reject videos shorter than min_duration, with an extreme aspect ratio, or that are vertical."""

    name = "duration_resolution"

    def __init__(self, min_duration: float = 10.0, max_aspect_ratio: float = 2.5):
        self.min_duration = min_duration
        self.max_aspect_ratio = max_aspect_ratio

    def __call__(self, video_path: str, frames: Optional[np.ndarray] = None) -> FilterResult:
        try:
            reader = imageio.get_reader(video_path, "ffmpeg")
            meta = reader.get_meta_data()
            w, h = meta["size"]
            duration = float(meta["duration"])
            reader.close()
        except Exception as e:
            return FilterResult(passed=False, metadata={"error": str(e)})

        is_landscape = w >= h
        aspect_ratio = w / h if h > 0 else float("inf")

        return FilterResult(
            passed=(duration >= self.min_duration) and (aspect_ratio <= self.max_aspect_ratio) and is_landscape,
            scores={"duration_sec": round(duration, 3), "aspect_ratio": round(aspect_ratio, 3)},
            metadata={"width": w, "height": h},
        )
