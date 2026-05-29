from typing import Optional

import cv2
import numpy as np
from PIL import Image


def sample_frames(video_path: str, n_frames: int = 8) -> Optional[np.ndarray]:
    """Sample n_frames evenly spaced from a video.

    Returns:
        (N, H, W, 3) uint8 RGB array, or None if the video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None

    n = min(n_frames, total)
    indices = np.linspace(0, total - 1, n, dtype=int)

    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return np.array(frames, dtype=np.uint8) if frames else None


def to_pil(frames: np.ndarray) -> list[Image.Image]:
    """Convert (N, H, W, 3) uint8 array to a list of PIL Images."""
    return [Image.fromarray(f) for f in frames]
