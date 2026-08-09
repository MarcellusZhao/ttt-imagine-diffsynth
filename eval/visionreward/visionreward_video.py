"""
VisionReward-Video scorer — a corrected, batched port of THUDM/VisionReward's
`inference-video.py` (https://github.com/THUDM/VisionReward, arXiv:2412.21059).

VisionReward is a checklist reward model: it asks a CogVLM2-video backbone a fixed
list of yes/no questions about a clip (some of which interpolate the generation
prompt), maps each answer to +1/-1, and returns the weighted mean under a learned
weight vector. Questions live in `assets/VisionReward_video_qa_select.txt` (29 of
them) and weights in `assets/weight.json` (29 floats, positionally aligned) — both
vendored verbatim from the upstream repo. The 64-question `VisionReward_video_qa.txt`
is vendored too but has NO published weight vector for video, so it cannot be scored;
it is here only for reference.

Differences from upstream `inference-video.py`, all of them fixes:

  * **Yes/no parsing.** Upstream compares the decoded token to the lowercase string
    `'yes'`, but the model emits `'Yes'` — so every answer scored as -1 and every
    video got the same (wrong) score, `mean(-weight)`. We compare case-insensitively.
  * **Decode once per clip.** Upstream re-reads and re-decodes the .mp4 inside
    `inference()`, i.e. once per question — 29 full decodes per clip.
  * **Encode the video once per clip.** All 29 questions share one video, so the
    63-layer EVA2-CLIP vision tower is run 29 times on identical pixels. We wrap
    `model.model.encode_images` with a one-entry cache keyed on tensor identity,
    which is the single largest speedup here.
  * **One decoding step, not 2048.** Upstream calls `generate(max_new_tokens=2048)`
    and then keeps only the first new token. We take the argmax of a single forward
    pass, which is identical to greedy decoding's first token.
  * **Optional question batching** (`batch_size > 1`): the questions differ only in
    text, so they can be right-padded into one batch sharing one vision encode. Right
    padding is safe because we never generate past the first token and attention is
    causal. Off by default; `--verify_batching` in the driver checks a batched pass
    against the batch-1 path on a real clip.

Frame sampling follows upstream's `chat` strategy: one frame per second, capped at
`num_frames` (24). **On clips longer than 24s that means only the first 24 seconds
are seen.** For long-video arms use `frame_sampling="uniform"`, which spreads the
same 24 frames over the whole clip. Keep the choice fixed across arms.
"""

import json
import os
import sys
from typing import List, Optional, Sequence

import numpy as np
import torch

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
DEFAULT_QUESTIONS_PATH = os.path.join(ASSET_DIR, "VisionReward_video_qa_select.txt")
DEFAULT_WEIGHT_PATH = os.path.join(ASSET_DIR, "weight.json")
DEFAULT_MODEL_PATH = os.environ.get(
    "VISIONREWARD_MODEL_PATH", "/work/nlp/hzhao/checkpoints/visionreward/VisionReward-Video"
)

PROMPT_PLACEHOLDER = "[[prompt]]"
NUM_FRAMES = 24
PAD_TOKEN_ID = 128002


# ──────────────────────────────────────────────────────────────────────────────
# Dependency shims
#
# The CogVLM2 remote code imports `pytorchvideo.transforms.ShortSideScale` and
# `torchvision.transforms._transforms_video` at module scope. pytorchvideo is
# unmaintained (last release 2022) and fails to import against modern torchvision,
# and `_transforms_video` is deprecated. Rather than pin the whole env to 2022, we
# install minimal stand-ins for exactly the three transforms the remote code uses.
# Each is a faithful reimplementation of the upstream op; if the real package
# imports cleanly we leave it alone.
# ──────────────────────────────────────────────────────────────────────────────

def install_video_transform_shims(verbose: bool = True) -> None:
    """Make the CogVLM2 remote code importable without pytorchvideo."""
    import types

    def _log(msg):
        if verbose:
            print(f"[VisionReward] {msg}")

    try:  # torchvision.transforms._transforms_video — deprecated, may be gone.
        from torchvision.transforms import _transforms_video  # noqa: F401
    except Exception:
        import torchvision.transforms as tvt

        class NormalizeVideo(torch.nn.Module):
            """(C, T, H, W) normalize — upstream torchvision semantics."""

            def __init__(self, mean, std, inplace=False):
                super().__init__()
                self.mean, self.std, self.inplace = mean, std, inplace

            def forward(self, clip):
                # Normalize over C by moving the channel axis to the front of a
                # (C, T, H, W) tensor: functional.normalize expects (..., C, H, W).
                return tvt.functional.normalize(
                    clip.permute(1, 0, 2, 3), self.mean, self.std, self.inplace
                ).permute(1, 0, 2, 3)

        class CenterCropVideo(torch.nn.Module):
            def __init__(self, crop_size):
                super().__init__()
                self.crop_size = (
                    (int(crop_size), int(crop_size))
                    if isinstance(crop_size, (int, float))
                    else tuple(crop_size)
                )

            def forward(self, clip):
                return tvt.functional.center_crop(clip, list(self.crop_size))

        mod = types.ModuleType("torchvision.transforms._transforms_video")
        mod.NormalizeVideo = NormalizeVideo
        mod.CenterCropVideo = CenterCropVideo
        sys.modules["torchvision.transforms._transforms_video"] = mod
        _log("shimmed torchvision.transforms._transforms_video")

    try:
        from pytorchvideo.transforms import ShortSideScale  # noqa: F401
        return
    except Exception:
        pass

    class ShortSideScale(torch.nn.Module):
        """Resize a (C, T, H, W) clip so that min(H, W) == size, preserving aspect.

        Mirrors pytorchvideo.transforms.ShortSideScale: bilinear, align_corners=False,
        antialias off — matching the interpolation VisionReward was trained with.
        """

        def __init__(self, size, interpolation="bilinear"):
            super().__init__()
            self.size, self.interpolation = int(size), interpolation

        def forward(self, clip):
            c, t, h, w = clip.shape
            if w < h:
                new_w, new_h = self.size, int(round(float(h) / w * self.size))
            else:
                new_h, new_w = self.size, int(round(float(w) / h * self.size))
            return torch.nn.functional.interpolate(
                clip, size=(new_h, new_w), mode=self.interpolation, align_corners=False
            )

    pkg = types.ModuleType("pytorchvideo")
    transforms_mod = types.ModuleType("pytorchvideo.transforms")
    transforms_mod.ShortSideScale = ShortSideScale
    pkg.transforms = transforms_mod
    sys.modules.setdefault("pytorchvideo", pkg)
    sys.modules["pytorchvideo.transforms"] = transforms_mod
    _log("shimmed pytorchvideo.transforms.ShortSideScale")


# ──────────────────────────────────────────────────────────────────────────────
# Checklist assets
# ──────────────────────────────────────────────────────────────────────────────

def load_checklist(questions_path: str = DEFAULT_QUESTIONS_PATH,
                   weight_path: str = DEFAULT_WEIGHT_PATH):
    """Return (questions, weights) and assert they are positionally aligned."""
    with open(questions_path, "r") as f:
        questions = [line.strip() for line in f if line.strip()]
    with open(weight_path, "r") as f:
        weights = np.array(json.load(f), dtype=np.float64)
    if len(questions) != len(weights):
        raise ValueError(
            f"{len(questions)} questions in {questions_path} but {len(weights)} weights "
            f"in {weight_path}. The two files are positionally aligned, so a mismatch "
            f"means the wrong question file — video weights pair with "
            f"VisionReward_video_qa_select.txt (29 questions), not the 64-question set."
        )
    return questions, weights


# ──────────────────────────────────────────────────────────────────────────────
# Frame sampling
# ──────────────────────────────────────────────────────────────────────────────

def load_video_frames(video_path: str, strategy: str = "chat",
                      num_frames: int = NUM_FRAMES):
    """Decode `video_path` to a uint8 (C, T, H, W) tensor of `num_frames` frames.

    strategy:
      "chat"    — upstream default: the frame nearest each whole second, stopping at
                  `num_frames`. Sees only the FIRST `num_frames` seconds of the clip.
      "uniform" — `num_frames` evenly spaced over the entire clip. Use for long-video
                  arms so the score reflects the whole clip, not just its opening.

    Returns (frames, num_source_frames, fps).
    """
    import io

    from decord import VideoReader, bridge, cpu

    bridge.set_bridge("torch")
    with open(video_path, "rb") as f:
        reader = VideoReader(io.BytesIO(f.read()), ctx=cpu(0))

    total_frames = len(reader)
    fps = float(reader.get_avg_fps())

    if strategy == "uniform":
        frame_id_list = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    elif strategy == "chat":
        timestamps = [t[0] for t in reader.get_frame_timestamp(np.arange(total_frames))]
        max_second = round(max(timestamps)) + 1
        frame_id_list = []
        for second in range(max_second):
            closest = min(timestamps, key=lambda x: abs(x - second))
            frame_id_list.append(timestamps.index(closest))
            if len(frame_id_list) >= num_frames:
                break
    else:
        raise ValueError(f"unknown frame sampling strategy '{strategy}'")

    frames = reader.get_batch(frame_id_list).permute(3, 0, 1, 2)  # (C, T, H, W)
    return frames, total_frames, fps


# ──────────────────────────────────────────────────────────────────────────────
# Model wrapper
# ──────────────────────────────────────────────────────────────────────────────

class VisionRewardVideo:
    """VisionReward-Video: checklist VQA over a clip, reduced to one weighted score."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, device: str = "cuda",
                 dtype: Optional[torch.dtype] = None,
                 questions_path: str = DEFAULT_QUESTIONS_PATH,
                 weight_path: str = DEFAULT_WEIGHT_PATH,
                 frame_sampling: str = "chat", num_frames: int = NUM_FRAMES,
                 verbose: bool = True):
        install_video_transform_shims(verbose=verbose)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.frame_sampling = frame_sampling
        self.num_frames = num_frames
        self.questions, self.weights = load_checklist(questions_path, weight_path)

        if dtype is None:
            dtype = (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
                else torch.float16
            )
        self.dtype = dtype

        if verbose:
            print(f"[VisionReward] loading {model_path} ({dtype}) on {device} …")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.pad_token_id = PAD_TOKEN_ID
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=dtype, trust_remote_code=True
            )
            .eval()
            .to(device)
        )

        # "Yes"/"No" as the model actually emits them (leading-space variants too, so a
        # tokenizer that prefixes whitespace still resolves).
        self.yes_ids = self._answer_token_ids(["Yes", "yes", " Yes", " yes"])
        self.no_ids = self._answer_token_ids(["No", "no", " No", " no"])
        self._install_vision_cache()

    def _answer_token_ids(self, variants: Sequence[str]) -> List[int]:
        ids = []
        for text in variants:
            encoded = self.tokenizer.encode(text, add_special_tokens=False)
            if encoded:
                ids.append(encoded[0])
        return sorted(set(ids))

    def _install_vision_cache(self):
        """Encode a clip's pixels once, reuse across the whole checklist.

        `CogVLMVideoModel.encode_images` reads only `images[0][0]` and returns
        (T, tokens_per_frame, d) — dim 0 is FRAMES, not batch; the caller flattens it
        with `rearrange(..., 'b n d -> (b n) d')` into this sample's vision tokens.
        All questions for a clip pass the *same* tensor object, so we cache on `id()`
        (one entry, replaced when a new clip comes in).

        For a batch of B identical-video rows the caller needs B*T*tokens_per_frame
        rows in row-major order, i.e. the per-sample block repeated B times — so
        `repeat` along dim 0, not `expand`, which would wrongly reinterpret dim 0 as
        the batch axis.
        """
        inner = self.model.model
        original = inner.encode_images
        cache = {"key": None, "value": None}

        def encode_images(images):
            video = images[0][0]
            key = id(video)
            if cache["key"] != key:
                cache["key"] = key
                cache["value"] = original([[video]])
            feats = cache["value"]
            batch = len(images)
            return feats if batch == 1 else feats.repeat(batch, 1, 1)

        inner.encode_images = encode_images
        self._vision_cache = cache

    # ── inference ─────────────────────────────────────────────────────────────

    def _build_inputs(self, query: str, video: torch.Tensor):
        return self.model.build_conversation_input_ids(
            tokenizer=self.tokenizer, query=query, images=[video],
            history=[], template_version="chat",
        )

    @torch.no_grad()
    def ask(self, video: torch.Tensor, queries: Sequence[str],
            batch_size: int = 1) -> List[str]:
        """Answer each query about `video`; returns "yes"/"no" per query."""
        answers: List[str] = []
        for start in range(0, len(queries), batch_size):
            chunk = queries[start:start + batch_size]
            built = [self._build_inputs(q, video) for q in chunk]
            answers.extend(self._forward_batch(built))
        return answers

    def _forward_batch(self, built) -> List[str]:
        lengths = [b["input_ids"].shape[0] for b in built]
        max_len = max(lengths)
        device = self.device

        # Right padding: we read logits at each row's own last real position and only
        # ever take one step, and attention is causal, so trailing pads cannot affect
        # any real position. (Left padding would shift position_ids, which the remote
        # code derives from token_type_ids.)
        def pad(seq, value):
            out = torch.full((len(built), max_len), value, dtype=torch.long)
            for i, b in enumerate(built):
                out[i, :lengths[i]] = seq(b)
            return out.to(device)

        input_ids = pad(lambda b: b["input_ids"], PAD_TOKEN_ID)
        token_type_ids = pad(lambda b: b["token_type_ids"], 0)  # 0 == LANGUAGE
        attention_mask = pad(lambda b: b["attention_mask"], 0)
        images = [[b["images"][0].to(device).to(self.dtype)] for b in built]

        logits = self.model(
            input_ids=input_ids, token_type_ids=token_type_ids,
            attention_mask=attention_mask, images=images, use_cache=False,
        ).logits

        answers = []
        for i, length in enumerate(lengths):
            last = logits[i, length - 1].float()
            top = int(torch.argmax(last).item())
            if top in self.yes_ids:
                answers.append("yes")
            elif top in self.no_ids:
                answers.append("no")
            else:
                # The checklist is yes/no by construction; if greedy decoding wanders
                # off it, fall back to the binary comparison the checklist implies
                # rather than silently scoring the answer as "no".
                yes = last[self.yes_ids].max()
                no = last[self.no_ids].max()
                answers.append("yes" if yes >= no else "no")
        return answers

    def score_video(self, video_path: str, prompt: str, batch_size: int = 1):
        """Score one clip. Returns a dict with the score and the raw answer vector."""
        frames, total_frames, fps = load_video_frames(
            video_path, strategy=self.frame_sampling, num_frames=self.num_frames
        )
        queries = [q.replace(PROMPT_PLACEHOLDER, prompt) for q in self.questions]
        answers = self.ask(frames, queries, batch_size=batch_size)
        signs = np.array([1.0 if a == "yes" else -1.0 for a in answers])
        duration = total_frames / fps if fps else float("nan")
        return {
            "score": float(np.mean(signs * self.weights)),
            "answers": answers,
            "num_yes": int(sum(a == "yes" for a in answers)),
            "num_questions": len(answers),
            "source_frames": int(total_frames),
            "fps": round(fps, 3),
            "duration_sec": round(duration, 2),
            "frames_seen": int(frames.shape[1]),
            "frame_sampling": self.frame_sampling,
            # True when "chat" sampling stopped before the clip ended, i.e. the tail
            # of the video was never shown to the reward model.
            "truncated": bool(
                self.frame_sampling == "chat" and duration > self.num_frames + 1
            ),
        }
