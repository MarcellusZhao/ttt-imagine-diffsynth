"""Aesthetic quality filter using the LAION improved aesthetic predictor.

Model: CLIP ViT-L/14 (openai) + lightweight MLP head trained on aesthetic ratings.
MLP weights are downloaded on first use from the official LAION repository and
cached at ~/.cache/videotuna/aesthetic_mlp.pth.

Training Data Score range: 1–10. Typical threshold: 4.5 (default).
"""

import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .base import FilterResult, VideoFilter
from .utils import sample_frames, to_pil

_MLP_WEIGHTS_URL = (
    "https://github.com/christophschuhmann/improved-aesthetic-predictor"
    "/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
)
_MLP_CACHE = Path.home() / ".cache" / "videotuna" / "aesthetic_mlp.pth"
_CLIP_MODEL = "ViT-L-14"
_CLIP_PRETRAINED = "openai"


class _AestheticMLP(nn.Module):
    def __init__(self, input_size: int = 768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def _load_models(device: str):
    import open_clip

    if not _MLP_CACHE.exists():
        _MLP_CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading aesthetic MLP weights → {_MLP_CACHE}")
        urllib.request.urlretrieve(_MLP_WEIGHTS_URL, _MLP_CACHE)

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        _CLIP_MODEL, pretrained=_CLIP_PRETRAINED
    )
    clip_model = clip_model.to(device).eval()

    mlp = _AestheticMLP(768).to(device)
    state = torch.load(_MLP_CACHE, map_location=device, weights_only=True)
    mlp.load_state_dict(state)
    mlp.eval()

    return clip_model, preprocess, mlp


class AestheticFilter(VideoFilter):
    """Score visual quality on a 0–10 scale; reject videos below min_score."""

    name = "aesthetic"

    def __init__(self, min_score: float = 4.5, n_frames: int = 8, device: str = "cpu"):
        self.min_score = min_score
        self.n_frames = n_frames
        self.device = device
        self._models = None

    @property
    def models(self):
        if self._models is None:
            self._models = _load_models(self.device)
        return self._models

    def _score_pil(self, pil_images) -> list[float]:
        clip_model, preprocess, mlp = self.models
        batch = torch.stack([preprocess(img) for img in pil_images]).to(self.device)
        with torch.no_grad():
            feats = clip_model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            scores = mlp(feats.float()).squeeze(-1)
        return scores.cpu().tolist()

    def __call__(self, video_path: str, frames: Optional[np.ndarray] = None) -> FilterResult:
        if frames is None:
            frames = sample_frames(video_path, self.n_frames)
        if frames is None or len(frames) == 0:
            return FilterResult(passed=False, metadata={"error": "could not read frames"})

        scores = self._score_pil(to_pil(frames))
        avg = float(np.mean(scores))

        return FilterResult(
            passed=avg >= self.min_score,
            scores={"score": round(avg, 4)},
        )
