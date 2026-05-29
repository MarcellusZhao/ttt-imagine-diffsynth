"""NSFW content detection for video frames.

Model: Falconsai/nsfw_image_detection (ViT image classifier, HuggingFace)
Strategy: score N sampled frames; fail if the max per-frame NSFW score exceeds threshold.
Using max rather than mean is intentional — a single NSFW frame is enough to reject.
"""

from typing import Optional

import numpy as np

from .base import FilterResult, VideoFilter
from .utils import sample_frames, to_pil

_MODEL_ID = "Falconsai/nsfw_image_detection"


class NSFWFilter(VideoFilter):
    """Reject videos containing NSFW content via image classification."""

    name = "nsfw"

    def __init__(
        self,
        max_score: float = 0.2,
        n_frames: int = 8,
        device: str = "cpu",
    ):
        self.max_score = max_score
        self.n_frames = n_frames
        self.device = device
        self._classifier = None

    @property
    def classifier(self):
        if self._classifier is None:
            from transformers import pipeline

            self._classifier = pipeline(
                "image-classification",
                model=_MODEL_ID,
                device=self.device,
            )
        return self._classifier

    def __call__(self, video_path: str, frames: Optional[np.ndarray] = None) -> FilterResult:
        if frames is None:
            frames = sample_frames(video_path, self.n_frames)
        if frames is None or len(frames) == 0:
            return FilterResult(passed=False, metadata={"error": "could not read frames"})

        pil_frames = to_pil(frames)
        results = self.classifier(pil_frames)

        nsfw_scores = []
        for frame_result in results:
            score = next(
                (item["score"] for item in frame_result if item["label"].lower() == "nsfw"),
                0.0,
            )
            nsfw_scores.append(score)

        max_score = float(np.max(nsfw_scores))
        mean_score = float(np.mean(nsfw_scores))

        return FilterResult(
            passed=max_score <= self.max_score,
            scores={
                "max_score": round(max_score, 4),
                "mean_score": round(mean_score, 4),
            },
        )
