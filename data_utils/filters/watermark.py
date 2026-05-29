"""Watermark and logo detection using CLIP zero-shot classification.

Strategy:
  1. Score full frames for global watermark presence.
  2. Score each corner crop independently to localise the watermark.
  3. Emit a crop_box that excludes detected watermark corners, so the
     training data loader can trim them at read time.

Model: CLIP ViT-B/32 (openai) via open_clip — already a project dependency.
"""

from typing import Optional

import numpy as np
from PIL import Image

from .base import FilterResult, VideoFilter
from .utils import sample_frames, to_pil

_CLIP_MODEL = "ViT-B-32"
_CLIP_PRETRAINED = "openai"

_WATERMARK_TEXTS = [
    "a photo with a visible watermark or logo overlay",
    "a clean photo without any watermark or logo",
]

# Indices into the four corners: top-left, top-right, bottom-left, bottom-right
_CORNER_NAMES = ["top_left", "top_right", "bottom_left", "bottom_right"]


def _corner_crops(frame: np.ndarray, fraction: float) -> list[Image.Image]:
    h, w = frame.shape[:2]
    ch, cw = max(1, int(h * fraction)), max(1, int(w * fraction))
    return [
        Image.fromarray(frame[:ch, :cw]),       # top-left
        Image.fromarray(frame[:ch, w - cw:]),   # top-right
        Image.fromarray(frame[h - ch:, :cw]),   # bottom-left
        Image.fromarray(frame[h - ch:, w - cw:]),  # bottom-right
    ]


def _watermark_crop_box(
    w: int,
    h: int,
    corner_probs: list[float],
    fraction: float,
    threshold: float,
) -> Optional[list[int]]:
    """Return [x1, y1, x2, y2] that crops out detected watermark corners."""
    ch, cw = int(h * fraction), int(w * fraction)
    x1, y1, x2, y2 = 0, 0, w, h
    # top-left
    if corner_probs[0] > threshold:
        x1 = max(x1, cw)
        y1 = max(y1, ch)
    # top-right
    if corner_probs[1] > threshold:
        x2 = min(x2, w - cw)
        y1 = max(y1, ch)
    # bottom-left
    if corner_probs[2] > threshold:
        x1 = max(x1, cw)
        y2 = min(y2, h - ch)
    # bottom-right
    if corner_probs[3] > threshold:
        x2 = min(x2, w - cw)
        y2 = min(y2, h - ch)
    if x1 == 0 and y1 == 0 and x2 == w and y2 == h:
        return None
    return [x1, y1, x2, y2]


class WatermarkFilter(VideoFilter):
    """Detect watermarks/logos via CLIP zero-shot; emit crop box for localised marks."""

    name = "watermark"

    def __init__(
        self,
        max_score: float = 0.5,
        n_frames: int = 4,
        corner_fraction: float = 0.15,
        device: str = "cpu",
    ):
        self.max_score = max_score
        self.n_frames = n_frames
        self.corner_fraction = corner_fraction
        self.device = device
        self._models = None

    @property
    def models(self):
        if self._models is None:
            import open_clip
            import torch

            model, _, preprocess = open_clip.create_model_and_transforms(
                _CLIP_MODEL, pretrained=_CLIP_PRETRAINED
            )
            model = model.to(self.device).eval()
            tokenizer = open_clip.get_tokenizer(_CLIP_MODEL)
            tokens = tokenizer(_WATERMARK_TEXTS)

            with torch.no_grad():
                text_feats = model.encode_text(tokens.to(self.device))
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

            self._models = (model, preprocess, text_feats)
        return self._models

    def _watermark_probs(self, pil_images: list[Image.Image]) -> list[float]:
        import torch

        model, preprocess, text_feats = self.models
        batch = torch.stack([preprocess(img) for img in pil_images]).to(self.device)
        with torch.no_grad():
            img_feats = model.encode_image(batch)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            # logit scale is temperature; default open_clip logit_scale ≈ 100
            logits = (img_feats @ text_feats.T) * model.logit_scale.exp()
            probs = logits.softmax(dim=-1)[:, 0]  # prob of "with watermark"
        return probs.cpu().tolist()

    def __call__(self, video_path: str, frames: Optional[np.ndarray] = None) -> FilterResult:
        if frames is None:
            frames = sample_frames(video_path, self.n_frames)
        if frames is None or len(frames) == 0:
            return FilterResult(passed=False, metadata={"error": "could not read frames"})

        global_score = float(np.mean(self._watermark_probs(to_pil(frames))))

        # Corner analysis on the middle frame for localisation
        mid = frames[len(frames) // 2]
        corners = _corner_crops(mid, self.corner_fraction)
        corner_probs = self._watermark_probs(corners)

        detected_corners = [_CORNER_NAMES[i] for i, p in enumerate(corner_probs) if p > self.max_score]
        h, w = mid.shape[:2]
        crop_box = _watermark_crop_box(w, h, corner_probs, self.corner_fraction, self.max_score)

        return FilterResult(
            passed=global_score <= self.max_score,
            scores={"score": round(global_score, 4)},
            metadata={
                "detected_corners": detected_corners,
                "crop_box": crop_box,
            },
        )
