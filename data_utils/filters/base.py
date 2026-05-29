from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class FilterResult:
    passed: bool
    scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, prefix: str = "") -> dict[str, Any]:
        out: dict[str, Any] = {f"{prefix}pass": self.passed}
        out.update({f"{prefix}{k}": v for k, v in self.scores.items()})
        out.update({f"{prefix}{k}": v for k, v in self.metadata.items()})
        return out


class VideoFilter(ABC):
    name: str = "base"
    dry_run: bool = False  # set to True by the pipeline; filters with side-effects must honour it

    @abstractmethod
    def __call__(self, video_path: str, frames: Optional[np.ndarray] = None) -> FilterResult:
        """
        Args:
            video_path: absolute path to the video file
            frames: optional pre-sampled RGB frames as (N, H, W, 3) uint8 array;
                    if None, the filter samples its own frames from video_path
        Returns:
            FilterResult with passed flag, numeric scores, and optional metadata
        """
        ...
