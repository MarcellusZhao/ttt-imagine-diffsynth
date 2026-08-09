"""
End-to-End Test-Time Training (E2E-TTT) for long-context video generation.

This module is a DiffSynth-Studio-native port of the reference implementation that
lives in the sibling ``ttt-imagine`` (VideoTuna) repo under ``e2e_ttt_video/``. The
*algorithm* is identical; only the plumbing is adapted to DiffSynth conventions:

  * The rectified-flow objective is computed with DiffSynth's own
    ``FlowMatchScheduler`` ("Wan" template) and ``pipe.model_fn`` (``model_fn_wan_video``)
    instead of a hand-rolled velocity formula, so the TTT loss matches the loss the
    model was trained with (``FlowMatchSFTLoss``).
  * Meta-learning differentiates *through* the inner-loop LoRA updates by temporarily
    re-parameterising the DiT with ``torch.nn.utils.stateless`` so that the second-order
    meta-gradient flows back to the LoRA meta-initialisation phi_0.
  * LoRA is the PEFT adapter DiffSynth already injects (param names contain ``lora``).

Two execution modes (mirroring the reference):

  * **Meta-training** (``run_meta_inner_loop``): MAML-style outer loop over a
    temporally-ordered chunk sequence from one video. For each chunk k the inner loop
    *memorises* chunk k with differentiable SGD step(s) on the LoRA scratchpad
    (phi_k -> phi_{k+1}), then *predicts* the next chunk k+1 with the updated LoRA. The
    meta-loss = mean next-chunk prediction loss, kept second-order so it back-propagates
    through the inner updates to phi_0. ``write_back=False`` leaves the real LoRA leaves
    at phi_0, so the outer AdamW updates phi_0 directly (no restore step needed).

  * **Test-time sequential generation** (``WanE2ETTTSequentialGenerator``): generate one
    chunk, *memorise* it with in-place first-order LoRA updates, generate the next chunk
    with the adapted LoRA. The scratchpad is reset to phi_0 before each narrative.

Targets ``Wan2.1-T2V-1.3B`` and ``Wan2.2-TI2V-5B`` (single-DiT Wan pipelines).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # context manager used for differentiable parameter override
    from torch.nn.utils.stateless import _reparametrize_module as _reparametrize
except Exception:  # pragma: no cover - very old / future torch
    from torch.func import functional_call as _functional_call

    import contextlib

    @contextlib.contextmanager
    def _reparametrize(module, parameters_and_buffers, tie_weights=False, strict=False):
        # Fallback shim: functional_call only wraps a single module __call__, which is
        # not what we need (model_fn calls many sub-modules). Prefer the stateless
        # context manager above; this branch should essentially never run.
        raise RuntimeError(
            "torch.nn.utils.stateless._reparametrize_module is unavailable; "
            "E2E-TTT meta-training requires it."
        )

import contextlib

from .flow_match import FlowMatchScheduler


# Fused flash/sage attention kernels do not support the double-backward required by
# second-order meta-training. ``enable_double_backward_attention()`` disables them so the
# Wan DiT falls back to ``F.scaled_dot_product_attention``, which we then pin to the math
# backend (the only SDPA backend with reliable double-backward).
_FORCE_MATH_SDPA = False


def enable_double_backward_attention() -> None:
    """Disable fused (flash / sage) attention in the Wan DiT and force the math SDPA
    backend. Call once before second-order meta-training. No effect on first-order
    test-time TTT, where fused kernels are fine."""
    global _FORCE_MATH_SDPA
    from ..models import wan_video_dit as _dit

    for flag in ("FLASH_ATTN_3_AVAILABLE", "FLASH_ATTN_2_AVAILABLE", "SAGE_ATTN_AVAILABLE"):
        if hasattr(_dit, flag):
            setattr(_dit, flag, False)
    _FORCE_MATH_SDPA = True


@contextlib.contextmanager
def _maybe_math_sdpa():
    if not _FORCE_MATH_SDPA:
        yield
        return
    # Enter the math-SDPA context guard, but never wrap the `yield` in a
    # try/except that could swallow an exception thrown back in (e.g. the
    # _StopRecomputationError used by activation-checkpoint recompute) and
    # re-yield -- that raises "generator didn't stop after throw()".
    try:
        guard = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
        guard.__enter__()
    except Exception:
        guard = None
    try:
        yield
    finally:
        if guard is not None:
            guard.__exit__(None, None, None)


# --------------------------------------------------------------------------- #
# Config dataclasses                                                          #
# --------------------------------------------------------------------------- #


@dataclass
class InnerLoopConfig:
    """Inner-loop (memorize) knobs, shared by train-time and test-time."""

    num_gradient_steps: int = 1
    num_mc_samples: int = 4
    inner_lr_init: float = 5e-5
    max_inner_grad_norm: float = 1.0
    # Timestep sampling range as fractions of the 1000 training timesteps.
    min_timestep_boundary: float = 0.0
    max_timestep_boundary: float = 1.0
    # Optimizer choice for the differentiable inner update.
    optimizer: str = "sgd"  # one of: sgd, adamw, muon, muonclip
    # PERK-style meta-learned per-(param, step) learning rates (training only).
    meta_learn_lr: bool = False
    # Meta-learning algorithm (training only). One of:
    #   * "maml"    - exact second-order MAML (create_graph=True; needs double-backward
    #                 attention). Inner loop memorize chunk k -> predict chunk k+1.
    #   * "fomaml"  - first-order MAML: drop the Hessian (create_graph=False). Same
    #                 memorize->predict objective; grad reaches phi_0 only through the
    #                 clone->phi_0 identity path. Frees the graph each step; fused
    #                 attention OK; --e2e_truncate_steps is ignored.
    #   * "reptile" - plain SGD adaptation on the memorize chunks, then move phi_0 toward
    #                 the adapted weights. No predict term, no second-order graph.
    algorithm: str = "maml"
    # Back-compat alias: first_order=True selects FOMAML. Reconciled in __post_init__ so
    # `algorithm` is always the single source of truth; `first_order` ends up True for
    # any non-second-order algorithm (fomaml/reptile) for code/logs that still read it.
    first_order: bool = False

    def __post_init__(self):
        self.algorithm = str(self.algorithm).lower()
        if self.first_order and self.algorithm == "maml":
            self.algorithm = "fomaml"
        if self.algorithm not in ("maml", "fomaml", "reptile"):
            raise ValueError(
                f"InnerLoopConfig.algorithm must be one of maml|fomaml|reptile, "
                f"got {self.algorithm!r}"
            )
        self.first_order = self.algorithm != "maml"


@dataclass
class ChunkingConfig:
    """Temporal chunking for the memorize->predict inner loop.

    Each training video is split into contiguous sub-clips of ``frames_per_chunk``
    frames; each sub-clip is VAE-encoded independently (separate V-Tokens per chunk).
    ``frames_per_chunk`` should be 4n+1 so it aligns with the Wan temporal VAE
    compression.

    Only the chunk *size* is fixed; the chunk *count* adapts to each clip's length:
    a video of ``F`` frames yields ``F // frames_per_chunk`` chunks. ``num_chunks`` is
    an optional upper bound (``None`` = unbounded) kept purely as a memory ceiling,
    since the second-order MAML graph grows with the chunk count. A clip too short to
    form 2 chunks (the memorize->predict minimum) is skipped, not chunked.
    """

    frames_per_chunk: int = 21  # 4*5 + 1
    num_chunks: Optional[int] = None  # None = adaptive (one chunk per frames_per_chunk frames)


@dataclass
class InferenceConfig:
    """High-level sequential generation controls (test-time)."""

    num_chunks: int = 4
    frames_per_chunk: int = 49  # 4n+1
    ttt_steps_per_chunk: int = 1
    height: int = 480
    width: int = 832
    num_inference_steps: int = 50
    sigma_shift: float = 5.0
    cfg_scale: float = 5.0
    seed: int = 42
    tiled: bool = True
    # Autoregressive frame conditioning: condition each chunk on the last frame of
    # the previous one (via TI2V-5B's fused VAE first-frame latent). Complements the
    # LoRA memory scratchpad with an explicit pixel-space anchor. TI2V-5B only.
    condition_on_last_frame: bool = False
    # When conditioning on the last frame, the first frame of each follow-up chunk
    # reproduces that anchor frame; drop it to avoid a duplicate-frame seam.
    drop_boundary_frame: bool = True
    # First-frame "sink": additionally pin the video's very first generated frame
    # alongside the sliding condition_on_last_frame anchor, for every follow-up chunk.
    # On by default (matching training's --e2e_condition_on_sink_frame), but *gated* on
    # condition_on_last_frame above: with no local anchor there is nothing to pair the sink
    # with, so it stays inert. That gate is what lets this default to True harmlessly on
    # models without fused first-frame conditioning. TI2V-5B only.
    condition_on_first_frame_sink: bool = True
    # Number of LOCAL anchor latent frames taken from the preceding chunk (k). 1 = the
    # legacy single-frame anchor. k > 1 pins a contiguous *block* of the previous chunk's
    # tail, which is what actually carries velocity: a single frame is motion-ambiguous,
    # so the LoRA scratchpad is the only channel for "which way was it going". See
    # ``anchor_overlap_pixel_frames`` for the pixel-frame cost. TI2V-5B only.
    num_anchor_latent_frames: int = 1
    # Use the ATTENTION anchor instead of the fused first-frame one: the preceding chunk's
    # trailing ``num_anchor_latent_frames`` latents (plus the sink) are prepended to the token
    # sequence as extra clean positions rather than overwriting the chunk's own leading
    # latents. Works on any plain T2V DiT (Wan2.1-T2V-1.3B/14B), needs no VAE encode at test
    # time, keeps every generated latent supervised, and produces no boundary frame to trim.
    # Mutually exclusive with the fused path; see ``attention_anchor_latents``.
    attention_anchor: bool = False


def attention_anchor_latents(
    prev_chunk: Optional[torch.Tensor],
    num_anchor_latent_frames: int,
    sink_latent: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Build the E2E-TTT attention-anchor block for the chunk that follows ``prev_chunk``.

    ``[sink_latent] + prev_chunk[:, :, -k:]`` -- the video's first latent frame (fixed, if
    given) followed by the preceding chunk's trailing k latents (sliding). Returns ``None``
    when there is no predecessor, i.e. for chunk 0, which is generated unconditioned.

    Two properties make this valid on both sides of train/test:

    * The frames come from ONE contiguous encode (training: the VAE encode of that chunk;
      inference: the sampler's own output latents for that chunk), so the k latents are
      genuine positional latents rather than stacked single-frame encodes -- the same
      requirement the fused path documents at ``WanVideoUnit_ImageEmbedderFused``.
    * They are taken from the chunk *tail*, and the sink from position 0 of chunk 0, in both
      training and inference. Unlike the fused path there is no overlap arithmetic to keep in
      sync: chunks are contiguous and non-overlapping, and the anchor is simply the previous
      chunk's last k latents.

    A k-frame block (rather than 1) is what carries velocity across the boundary; with k=1 the
    anchor is motion-ambiguous and the LoRA scratchpad is the only channel for direction.
    """
    if prev_chunk is None:
        return None
    k = max(1, int(num_anchor_latent_frames))
    if prev_chunk.shape[2] < k:
        raise ValueError(
            f"attention anchor needs k={k} latent frames but the preceding chunk has only "
            f"{prev_chunk.shape[2]}; raise frames_per_chunk or lower num_anchor_latent_frames."
        )
    block = prev_chunk[:, :, -k:]
    if sink_latent is not None:
        block = torch.cat([sink_latent, block], dim=2)
    return block


def num_pinned_pixel_frames(num_clean_latent_frames: int) -> int:
    """Decoded frames covered by the leading ``num_clean_latent_frames`` pinned latents.

    The Wan temporal VAE maps ``T`` latent frames to ``4 * (T - 1) + 1`` pixel frames:
    latent frame 0 decodes to a single pixel frame, every later latent frame to 4. So a
    lone local anchor (1 clean latent) pins 1 decoded frame, while a first-frame sink at
    position 0 plus the local anchor at position 1 (2 clean latents) pins 1 + 4 = 5.
    Those decoded frames are given context, not generated content, and are what
    ``drop_boundary_frame`` trims at each chunk boundary.
    """
    n = max(1, int(num_clean_latent_frames))
    return 4 * (n - 1) + 1


def num_clean_latents(num_anchor_latent_frames: int, use_sink: bool) -> int:
    """Total pinned leading latent frames: the local anchor block plus the optional sink."""
    return max(1, int(num_anchor_latent_frames)) + (1 if use_sink else 0)


def anchor_overlap_pixel_frames(num_anchor_latent_frames: int, use_sink: bool) -> int:
    """Pixel frames of the preceding chunk that the anchor block consumes.

    This is both the chunk *overlap* meta-training must slice with and the number of
    decoded frames inference must hand forward, so training and test-time see byte-identical
    anchor latents.

    The Wan causal VAE gives latent 0 exactly 1 pixel frame and every later latent 4. With a
    sink the anchor block is displaced to latent positions ``1..k``, so it covers ``4k``
    pixel frames -- but producing latents that are *statistically correct at those positions*
    requires one extra leading pixel frame as causal context (whose latent 0 the sink then
    overwrites). Hence ``4k + 1 = num_pinned_pixel_frames(k + 1)``. Without a sink the block
    sits at ``0..k-1`` and covers ``num_pinned_pixel_frames(k)``.

    ``k == 1`` is the legacy single-frame anchor and always consumes exactly 1 pixel frame:
    there, the anchor is a standalone single-frame encode placed at latent 0 (and the sink,
    when present, is *prepended* rather than overwriting a contiguous encode). Keeping that
    case byte-identical is what lets phi_0 checkpoints trained before this flag existed still
    reproduce their baseline.
    """
    k = max(1, int(num_anchor_latent_frames))
    if k == 1:
        return 1
    return num_pinned_pixel_frames(num_clean_latents(k, use_sink))


# --------------------------------------------------------------------------- #
# Differentiable inner-loop optimizers (model-agnostic, direct port)          #
# --------------------------------------------------------------------------- #


def _per_tensor_clip(grad: torch.Tensor, max_grad_norm: float) -> torch.Tensor:
    if max_grad_norm <= 0:
        return grad
    gnorm = grad.detach().norm()
    if gnorm > max_grad_norm:
        grad = grad * (max_grad_norm / (gnorm + 1e-8))
    return grad


class DifferentiableSGD:
    """Manual differentiable SGD: ``w_new = w_old - lr * grad`` (graph-preserving)."""

    def __init__(self, lr: float = 1e-4, max_grad_norm: float = 1.0):
        self.lr = float(lr)
        self.max_grad_norm = float(max_grad_norm)

    def step(
        self,
        params: Dict[str, torch.Tensor],
        grads: Dict[str, Optional[torch.Tensor]],
        learned_lrs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        updated: Dict[str, torch.Tensor] = {}
        for name, param in params.items():
            grad = grads.get(name)
            if grad is None:
                updated[name] = param
                continue
            grad = _per_tensor_clip(grad, self.max_grad_norm)
            lr = self.lr
            if learned_lrs is not None and name in learned_lrs:
                lr = learned_lrs[name]
            updated[name] = param - lr * grad
        return updated


class DifferentiableAdamW:
    """Manual differentiable AdamW (decoupled weight decay), state kept in-instance."""

    def __init__(
        self,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
    ):
        self.lr = float(lr)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.max_grad_norm = float(max_grad_norm)
        self._step = 0
        self._m: Dict[str, torch.Tensor] = {}
        self._v: Dict[str, torch.Tensor] = {}

    def step(self, params, grads, learned_lrs=None):
        self._step += 1
        t = self._step
        updated: Dict[str, torch.Tensor] = {}
        for name, param in params.items():
            grad = grads.get(name)
            if grad is None:
                updated[name] = param
                continue
            grad = _per_tensor_clip(grad, self.max_grad_norm)
            m = self._m.get(name)
            v = self._v.get(name)
            if m is None:
                m = torch.zeros_like(param)
            if v is None:
                v = torch.zeros_like(param)
            m = self.beta1 * m + (1.0 - self.beta1) * grad
            v = self.beta2 * v + (1.0 - self.beta2) * (grad * grad)
            m_hat = m / (1.0 - (self.beta1 ** t))
            v_hat = v / (1.0 - (self.beta2 ** t))
            lr = self.lr
            if learned_lrs is not None and name in learned_lrs:
                lr = learned_lrs[name]
            if self.weight_decay != 0.0:
                param = param - lr * self.weight_decay * param
            param = param - lr * m_hat / (torch.sqrt(v_hat) + self.eps)
            self._m[name] = m
            self._v[name] = v
            updated[name] = param
        return updated


def _newton_schulz_orthogonalize(G: torch.Tensor, *, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    if G.ndim != 2:
        raise ValueError(f"Expected 2D tensor for Muon update, got shape {tuple(G.shape)}")
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.float32)
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(int(steps)):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + (B @ X)
    if transposed:
        X = X.T
    return X.to(dtype=G.dtype)


class DifferentiableMuon:
    """Differentiable Muon for 2D parameters: momentum SGD + Newton-Schulz orthogonalization."""

    def __init__(self, lr=1e-4, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0, max_grad_norm=1.0):
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)
        self.ns_steps = int(ns_steps)
        self.weight_decay = float(weight_decay)
        self.max_grad_norm = float(max_grad_norm)
        self._buf: Dict[str, torch.Tensor] = {}

    def step(self, params, grads, learned_lrs=None):
        updated: Dict[str, torch.Tensor] = {}
        beta = self.momentum
        for name, param in params.items():
            grad = grads.get(name)
            if grad is None:
                updated[name] = param
                continue
            if grad.ndim != 2:
                raise ValueError(f"Muon only supports 2D params; got {name} with grad shape {tuple(grad.shape)}")
            grad = _per_tensor_clip(grad, self.max_grad_norm)
            buf = self._buf.get(name)
            if buf is None:
                buf = torch.zeros_like(param)
            buf = beta * buf + (1.0 - beta) * grad
            update = grad + beta * (buf - grad) if self.nesterov else buf
            update = _newton_schulz_orthogonalize(update, steps=self.ns_steps)
            scale_factor = float(max(1.0, grad.size(-2) / grad.size(-1)) ** 0.5)
            update = update * scale_factor
            lr = self.lr
            if learned_lrs is not None and name in learned_lrs:
                lr = learned_lrs[name]
            if self.weight_decay != 0.0:
                param = param - lr * self.weight_decay * param
            param = param - lr * update
            self._buf[name] = buf
            updated[name] = param
        return updated


class DifferentiableMuonClip:
    """MuonClip: Muon for 2D params, AdamW for the rest, optional qk-clip rescaling."""

    def __init__(
        self,
        *,
        lr_muon=1e-4,
        lr_adamw=1e-4,
        muon_momentum=0.95,
        muon_nesterov=True,
        muon_ns_steps=5,
        adamw_betas=(0.9, 0.999),
        adamw_eps=1e-8,
        weight_decay=0.0,
        max_grad_norm=1.0,
        qk_clip_threshold: Optional[float] = None,
        qk_clip_alpha: float = 0.5,
        q_name_patterns: Sequence[str] = ("q_proj", "query", ".q.", ".q_"),
        k_name_patterns: Sequence[str] = ("k_proj", "key", ".k.", ".k_"),
    ):
        self.muon = DifferentiableMuon(
            lr=lr_muon, momentum=muon_momentum, nesterov=muon_nesterov,
            ns_steps=muon_ns_steps, weight_decay=weight_decay, max_grad_norm=max_grad_norm,
        )
        self.adamw = DifferentiableAdamW(
            lr=lr_adamw, betas=adamw_betas, eps=adamw_eps,
            weight_decay=weight_decay, max_grad_norm=max_grad_norm,
        )
        self.qk_clip_threshold = qk_clip_threshold
        self.qk_clip_alpha = float(qk_clip_alpha)
        self.q_name_patterns = tuple(q_name_patterns)
        self.k_name_patterns = tuple(k_name_patterns)

    def _is_query(self, name):
        n = name.lower()
        return any(pat in n for pat in self.q_name_patterns)

    def _is_key(self, name):
        n = name.lower()
        return any(pat in n for pat in self.k_name_patterns)

    def step(self, params, grads, learned_lrs=None, *, qk_clip_max_logit: Optional[torch.Tensor] = None):
        params_2d = {n: p for n, p in params.items() if p.ndim == 2}
        params_other = {n: p for n, p in params.items() if p.ndim != 2}
        grads_2d = {n: grads.get(n) for n in params_2d}
        grads_other = {n: grads.get(n) for n in params_other}
        learned_2d = learned_other = None
        if learned_lrs is not None:
            learned_2d = {n: learned_lrs[n] for n in params_2d if n in learned_lrs}
            learned_other = {n: learned_lrs[n] for n in params_other if n in learned_lrs}
        updated: Dict[str, torch.Tensor] = {}
        updated.update(self.muon.step(params_2d, grads_2d, learned_lrs=learned_2d))
        updated.update(self.adamw.step(params_other, grads_other, learned_lrs=learned_other))
        if self.qk_clip_threshold is not None and qk_clip_max_logit is not None:
            max_logit = float(qk_clip_max_logit.detach().item())
            if max_logit > 0:
                eta = min(self.qk_clip_threshold / max_logit, 1.0)
                if eta < 1.0:
                    alpha = self.qk_clip_alpha
                    scale_q, scale_k = eta ** alpha, eta ** (1.0 - alpha)
                    for n in list(updated.keys()):
                        if self._is_query(n):
                            updated[n] = updated[n] * scale_q
                        elif self._is_key(n):
                            updated[n] = updated[n] * scale_k
        return updated


class MetaLearnedLRSchedule(nn.Module):
    """PERK-style meta-learned per-parameter per-step learning rates (softplus-positive)."""

    def __init__(self, lora_param_names: List[str], num_inner_steps: int, init_lr: float = 5e-5):
        super().__init__()
        self.param_names = list(lora_param_names)
        self.num_steps = int(num_inner_steps)
        self.raw_lrs = nn.ParameterDict()
        init_lr_t = torch.tensor(float(init_lr))
        init_raw = torch.log(torch.exp(init_lr_t) - 1.0)
        for name in self.param_names:
            safe_name = name.replace(".", "_")
            for step in range(self.num_steps):
                self.raw_lrs[f"{safe_name}__step{step}"] = nn.Parameter(init_raw.clone())

    def get_lrs(self, step_index: int) -> Dict[str, torch.Tensor]:
        step_index = int(step_index)
        lrs: Dict[str, torch.Tensor] = {}
        for name in self.param_names:
            safe_name = name.replace(".", "_")
            raw = self.raw_lrs.get(f"{safe_name}__step{step_index}")
            if raw is None:
                lrs[name] = torch.tensor(5e-5, device=next(self.parameters()).device)
            else:
                lrs[name] = F.softplus(raw)
        return lrs


def make_inner_optimizer(inner_cfg: InnerLoopConfig):
    """Build a fresh differentiable inner-loop optimizer from the config."""
    opt = inner_cfg.optimizer.lower()
    if opt == "sgd":
        return DifferentiableSGD(lr=inner_cfg.inner_lr_init, max_grad_norm=inner_cfg.max_inner_grad_norm)
    if opt == "adamw":
        return DifferentiableAdamW(lr=inner_cfg.inner_lr_init, max_grad_norm=inner_cfg.max_inner_grad_norm)
    if opt == "muon":
        return DifferentiableMuon(lr=inner_cfg.inner_lr_init, max_grad_norm=inner_cfg.max_inner_grad_norm)
    if opt == "muonclip":
        return DifferentiableMuonClip(
            lr_muon=inner_cfg.inner_lr_init, lr_adamw=inner_cfg.inner_lr_init,
            max_grad_norm=inner_cfg.max_inner_grad_norm,
        )
    raise ValueError(f"Unknown inner-loop optimizer: {inner_cfg.optimizer!r}")


# --------------------------------------------------------------------------- #
# LoRA "memory scratchpad" helpers                                            #
# --------------------------------------------------------------------------- #


def get_lora_params(model: nn.Module) -> Dict[str, nn.Parameter]:
    """LoRA parameters (name -> Parameter) for the given module (PEFT naming)."""
    return {n: p for n, p in model.named_parameters() if "lora" in n}


def get_trainable_lora_params(model: nn.Module) -> Dict[str, nn.Parameter]:
    return {n: p for n, p in model.named_parameters() if "lora" in n and p.requires_grad}


def snapshot_lora_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in get_lora_params(model).items()}


def restore_lora_state(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    params = dict(model.named_parameters())
    with torch.no_grad():
        for n, v in state.items():
            if n in params:
                params[n].copy_(v)


def count_lora_params(model: nn.Module) -> int:
    return sum(p.numel() for p in get_lora_params(model).values())


# --------------------------------------------------------------------------- #
# DiffSynth-native rectified-flow loss + differentiable model_fn call         #
# --------------------------------------------------------------------------- #


def make_training_scheduler(sigma_shift: float = 5.0) -> FlowMatchScheduler:
    """A dedicated 1000-step training scheduler so we never clobber the pipeline's
    inference scheduler. Mirrors ``switch_pipe_to_training_mode``."""
    sched = FlowMatchScheduler("Wan")
    sched.set_timesteps(1000, training=True, shift=sigma_shift)
    return sched


# --------------------------------------------------------------------------- #
# Recycled-error buffers (anti-drift), following Stable-Video-Infinity          #
# --------------------------------------------------------------------------- #


class ErrorRecycler:
    """Recycled-error buffers following Stable-Video-Infinity (arXiv:2510.09212;
    reference implementation ``Stable-Video-Infinity/train_svi.py``), adapted to
    E2E-TTT meta-training.

    Anti-drift idea (SVI): at test time the model conditions on -- and, in E2E-TTT,
    *memorizes* -- its own imperfect outputs, but vanilla training only ever sees
    clean ground-truth chunks. This module harvests the model's real prediction
    errors during training into timestep-bucketed buffers and re-injects them into
    training inputs. How the target is treated depends on which objective is being
    corrupted -- this is where E2E-TTT departs from SVI's single training step:

      * MEMORIZE (inner/TTT) loss, gated by ``latent_prob``: the chunk is corrupted
        CONSISTENTLY (input and target), replicating the test-time
        ``ttt_update_inplace``, whose target is built from the generated --
        corrupted -- chunk (no clean data exists at inference). The anti-drift
        signal comes from the OUTER loop: phi_0 is meta-shaped so that memorizing
        corrupted chunks still predicts clean ones.
      * PREDICT (generation/meta) loss, gated by ``noise_prob``/``latent_prob``:
        SVI's asymmetric scheme -- corrupted inputs, CLEAN target -- teaching the
        adapted model to self-correct mid-trajectory sampling errors; plus the
        ``y_prob``-gated anchor-frame injection (SVI's y-error), simulating the
        drifted previous-chunk frame the anchor actually is at inference.

    Mirrors SVI's ``LightningModelForTrain_onestage`` buffer machinery:
      * two buffers, harvested via the two one-shot ``step(..., to_final=True)``
        extrapolations (SVI train_svi.py:1151-1160): ``noise_error_buffer`` holds
        noise-direction errors (SVI's ``latent_error_buffer``) and
        ``y_error_buffer`` holds data-direction errors (SVI's ``y_error_buffer``);
      * ``num_grids`` buckets keyed by the nearest timestep of a simulated
        ``num_grids``-step inference schedule (SVI's ``_get_timestep_grid``), so
        injected errors match the errors made at that point of a real sampling
        trajectory;
      * random-replacement eviction (SVI's default ``buffer_replacement_strategy``);
      * independent injection gates ``noise_prob`` / ``latent_prob`` / ``y_prob``
        plus a ``clean_prob`` override that forces a fully clean step;
      * intensity modulation by ``uniform(1-f, 1+f)`` (``error_modulate_factor``).

    Simplifications vs SVI (all safe to revisit later):
      * NO ``all_gather`` warmup -- buffers are per-process and filled from local
        errors only. ``buffer_warmup_iter`` instead gates when *injection* starts,
        so the first few outer steps only collect. E2E-TTT harvests
        ``num_mem_steps x num_mc_samples`` errors per outer step (vs SVI's 1), so
        local buckets fill quickly without cross-GPU sync.
      * only the random replacement strategy (SVI's default and fastest).

    CPU memory = 2 buffers x num_grids x error_buffer_k x one chunk-latent tensor
    (e.g. ~2 MB for a TI2V-5B 21-frame 704x1280 chunk -> ~6 GB at 50 x 32).
    """

    def __init__(
        self,
        num_grids: int = 50,
        error_buffer_k: int = 32,
        buffer_warmup_iter: int = 20,
        noise_prob: float = 0.9,
        latent_prob: float = 0.9,
        y_prob: float = 0.9,
        clean_prob: float = 0.1,
        error_modulate_factor: float = 0.0,
        sigma_shift: float = 5.0,
        anchor_sample_from_all_grids: bool = True,
    ):
        # Simulated inference schedule: one grid per inference timestep (SVI builds
        # this with get_timesteps(num_inference_steps=num_grids, shift=5.0)).
        sigmas, timesteps = FlowMatchScheduler.set_timesteps_wan(
            num_inference_steps=int(num_grids), shift=float(sigma_shift),
        )
        self.inference_sigmas = sigmas        # [num_grids], descending
        self.inference_timesteps = timesteps  # [num_grids], = sigmas * 1000
        self.error_buffer_size = int(error_buffer_k)
        self.buffer_warmup_iter = int(buffer_warmup_iter)
        self.noise_prob = float(noise_prob)
        self.latent_prob = float(latent_prob)
        self.y_prob = float(y_prob)
        self.clean_prob = float(clean_prob)
        self.error_modulate_factor = float(error_modulate_factor)
        num_grids = len(timesteps)
        # SVI naming note: SVI's `latent_error_buffer` stores the noise-direction
        # errors and feeds the *noise* injection; its `y_error_buffer` stores the
        # data-direction errors and feeds both the *latent* and *y* injections.
        # We name the first one `noise_error_buffer` for clarity.
        self.noise_error_buffer: Dict[int, List[torch.Tensor]] = {i: [] for i in range(num_grids)}
        self.y_error_buffer: Dict[int, List[torch.Tensor]] = {i: [] for i in range(num_grids)}
        # Dedicated anchor-error buffer. At inference the anchor is the previous chunk's
        # *last k generated frames*, so its drift is a terminal, data-space error of that
        # k-frame block -- a different distribution from the mid-trajectory, full-chunk data
        # errors in `y_error_buffer`. We harvest it separately from the predict step's LAST k
        # frames (the frames that become the next anchor) and inject the block into the pinned
        # anchor. Kept apart from the full-chunk buffers so its shape ([B,C,k,H,W]) never
        # collides with the full-chunk latent injections.
        self.anchor_error_buffer: Dict[int, List[torch.Tensor]] = {i: [] for i in range(num_grids)}
        # Anchor drift is terminal (data-space, timestep-independent), so it is not tied to
        # the current denoising timestep: sampling from any grid maximises buffer use.
        self.anchor_sample_from_all_grids = bool(anchor_sample_from_all_grids)
        self.iteration_count = 0
        self.num_harvested = 0
        # --- diagnostics surfaced via stats() (all cheap running scalars) ---
        # #1 harvested error RMS (per-element root-mean-square, comparable across the
        #    different-size buffers) accumulated per buffer type.
        self._harv_rms_sum = {"noise": 0.0, "y": 0.0, "anchor": 0.0}
        self._harv_rms_cnt = {"noise": 0, "y": 0, "anchor": 0}
        # #2 injection attempts vs successes per type -- a miss (empty grid or shape
        #    mismatch) silently injects nothing, so miss_rate flags anti-drift going inert.
        self._inj_req = {"noise": 0, "y": 0, "anchor": 0}
        self._inj_ok = {"noise": 0, "y": 0, "anchor": 0}
        # #3 relative corruption strength ||err|| / ||x0|| per type (summed over successes).
        self._inj_rel_sum = {"noise": 0.0, "y": 0.0, "anchor": 0.0}
        # #4 harvest events split by objective (confirms predict-side harvesting fires).
        self.num_harvested_memorize = 0
        self.num_harvested_predict = 0

    def begin_iteration(self) -> None:
        """Call once per outer training step; drives the injection warmup gate."""
        self.iteration_count += 1

    @property
    def inject_active(self) -> bool:
        """Injection starts only after the warmup iterations (collection-only phase)."""
        return self.iteration_count > self.buffer_warmup_iter

    def _get_timestep_grid(self, timestep) -> int:
        """Grid index for a timestep: nearest inference timestep (SVI's version)."""
        if isinstance(timestep, torch.Tensor):
            timestep_val = timestep.flatten()[0].item()
        else:
            timestep_val = float(timestep)
        timestep_val = max(0.0, min(timestep_val, 999.0))
        grid_idx = torch.argmin((self.inference_timesteps - timestep_val).abs()).item()
        return int(grid_idx)

    def _add_error_to_buffer(self, buffer: Dict[int, List[torch.Tensor]], error_sample: torch.Tensor, timestep, tag: str) -> None:
        """Random-replacement insertion (SVI's `_add_error_to_latent_buffer`, 'random')."""
        grid_idx = self._get_timestep_grid(timestep)
        # #1 harvested error scale, as per-element RMS so noise/y/anchor are comparable.
        self._harv_rms_sum[tag] += float(error_sample.detach().float().pow(2).mean().sqrt())
        self._harv_rms_cnt[tag] += 1
        error_cpu = error_sample.detach().cpu()
        bucket = buffer[grid_idx]
        if len(bucket) < self.error_buffer_size:
            bucket.append(error_cpu)
        else:
            bucket[random.randint(0, len(bucket) - 1)] = error_cpu
        self.num_harvested += 1

    def _record_injection(self, tag: str, err: torch.Tensor, like: torch.Tensor) -> None:
        """Log a successful injection: count it (#2) and its relative strength (#3)."""
        self._inj_ok[tag] += 1
        denom = float(like.detach().float().norm()) + 1e-8
        self._inj_rel_sum[tag] += float(err.detach().float().norm()) / denom

    def _sample_error_from_buffer(self, buffer: Dict[int, List[torch.Tensor]], timestep, like: torch.Tensor, tag: str) -> Optional[torch.Tensor]:
        """Random sample from the current timestep grid, intensity-modulated
        (SVI's `_sample_noise_error_from_noise_buffer`). None if the grid is empty
        or the stored shape does not match (mixed-resolution datasets)."""
        self._inj_req[tag] += 1  # #2 count the attempt (miss = returns None below)
        grid_idx = self._get_timestep_grid(timestep)
        bucket = buffer[grid_idx]
        if not bucket:
            return None
        error_sample = random.choice(bucket)
        if error_sample.shape != like.shape:
            return None
        error_sample = error_sample.to(device=like.device, dtype=like.dtype)
        intensity_mod = random.uniform(1.0 - self.error_modulate_factor, 1.0 + self.error_modulate_factor)
        error_sample = error_sample * intensity_mod
        self._record_injection(tag, error_sample, like)
        return error_sample

    def _sample_error_from_any_grid(self, buffer: Dict[int, List[torch.Tensor]], like: torch.Tensor, tag: str) -> Optional[torch.Tensor]:
        """Like ``_sample_error_from_buffer`` but pooled across ALL non-empty grids
        (SVI's ``y_error_sample_from_all_grids``). Used for the anchor error, whose
        terminal/data-space drift is not tied to the current denoising timestep."""
        self._inj_req[tag] += 1  # #2 count the attempt (miss = returns None below)
        pooled = [s for bucket in buffer.values() for s in bucket if s.shape == like.shape]
        if not pooled:
            return None
        error_sample = random.choice(pooled).to(device=like.device, dtype=like.dtype)
        intensity_mod = random.uniform(1.0 - self.error_modulate_factor, 1.0 + self.error_modulate_factor)
        error_sample = error_sample * intensity_mod
        self._record_injection(tag, error_sample, like)
        return error_sample

    @torch.no_grad()
    def harvest(self, scheduler: FlowMatchScheduler, noise_pred: torch.Tensor, training_target: torch.Tensor, noisy_latents: torch.Tensor, timestep, *, condition_on_first_frame: bool = False, num_clean_frames: int = 1, num_anchor_frames: int = 1, attention_anchor: bool = False, source: Optional[str] = None) -> None:
        """Harvest the model's one-shot extrapolation errors (SVI train_svi.py:1151-1160).

        SVI computes both errors with `scheduler.step(..., to_final=True, self_corr=...)`;
        DiffSynth's FlowMatchScheduler.step has no `self_corr` branch, so the two
        one-shot jumps are inlined here: self_corr=True ends at the noise end
        (sigma_=1), self_corr=False at the data end (sigma_=0).

        ``condition_on_first_frame`` (predict objective with the fused first-frame anchor):
        latent frames ``0:num_clean_frames`` are the pinned anchor(s) -- fed at timestep 0
        as clean context, never denoised, and excluded from the loss -- so their slice of
        ``noisy_latents`` sits at sigma~=0, not the sampled ``sigma``, making their
        extrapolated "error" numerically meaningless (and possibly carrying the injected
        anchor y-error). We therefore zero those leading frames in the full-chunk errors,
        and route the model's genuine terminal error on the chunk's LAST
        ``num_anchor_frames`` frames -- exactly the frames that become the next chunk's
        anchor at inference -- into the dedicated ``anchor_error_buffer`` as one contiguous
        block. ``num_clean_frames`` counts all pinned leading frames (anchor block + optional
        sink); ``num_anchor_frames`` is the anchor block width k alone, since the sink is a
        fixed reference that never drifts and is never injected into.

        Harvesting the last k frames as a *block* (rather than k single frames) is what makes
        the injected anchor error temporally coherent -- see the injection site in
        ``compute_flow_matching_loss``. The block width also happens to line up with the
        pixel frames it represents: at k=3 the last 3 latents of a chunk cover its last 12
        pixel frames, which is precisely the window inference hands forward.
        """
        if source == "memorize":
            self.num_harvested_memorize += 1  # #4 objective split
        elif source == "predict":
            self.num_harvested_predict += 1
        t = timestep.cpu() if isinstance(timestep, torch.Tensor) else timestep
        timestep_id = torch.argmin((scheduler.timesteps - t).abs())
        sigma = float(scheduler.sigmas[timestep_id])
        noise_pred = noise_pred.detach()
        training_target = training_target.detach()
        noisy_latents = noisy_latents.detach()

        x_0_pred = noisy_latents + noise_pred * (1.0 - sigma)
        noise_corr_gt = noisy_latents + training_target * (1.0 - sigma)
        noise_error = x_0_pred - noise_corr_gt

        x_1_pred = noisy_latents + noise_pred * (0.0 - sigma)
        latent_corr_gt = noisy_latents + training_target * (0.0 - sigma)
        y_error = x_1_pred - latent_corr_gt

        k_anchor = max(1, int(num_anchor_frames))
        if attention_anchor and y_error.shape[2] >= k_anchor:
            # Attention anchor: nothing is pinned, so every frame's extrapolated error is
            # genuine and the full-chunk buffers stay untouched. The chunk's last k latents
            # still become the next chunk's anchor block verbatim, so their terminal error is
            # exactly the drift that block will carry -- harvest it as one coherent slice.
            self._add_error_to_buffer(self.anchor_error_buffer, y_error[:, :, -k_anchor:], timestep, "anchor")
        elif condition_on_first_frame and y_error.shape[2] >= num_clean_frames + k_anchor:
            # The last k frames' data-direction error is the terminal error of frames
            # generated from a (possibly drifted) context -- exactly the anchor drift. Taken
            # as one slice so the block stays temporally coherent, and guarded so it can never
            # overlap the pinned leading frames (whose "error" is meaningless, see above).
            self._add_error_to_buffer(self.anchor_error_buffer, y_error[:, :, -k_anchor:], timestep, "anchor")
            # The leading anchor frame(s) are pinned: their extrapolated error is garbage,
            # so keep the full-chunk buffers uniform in shape by zeroing them rather than
            # dropping them.
            noise_error = noise_error.clone(); noise_error[:, :, 0:num_clean_frames] = 0
            y_error = y_error.clone(); y_error[:, :, 0:num_clean_frames] = 0

        self._add_error_to_buffer(self.noise_error_buffer, noise_error, timestep, "noise")
        self._add_error_to_buffer(self.y_error_buffer, y_error, timestep, "y")

    @torch.no_grad()
    def sample_noise_error(self, timestep, like: torch.Tensor) -> Optional[torch.Tensor]:
        """Noise-direction error for `noise = noise + err` injection."""
        return self._sample_error_from_buffer(self.noise_error_buffer, timestep, like, "noise")

    @torch.no_grad()
    def sample_latent_error(self, timestep, like: torch.Tensor) -> Optional[torch.Tensor]:
        """Data-direction error for `latents = latents + err` injection (SVI samples
        this from the y_error buffer -- see `_sample_latent_error_from_latent_buffer`)."""
        return self._sample_error_from_buffer(self.y_error_buffer, timestep, like, "y")

    @torch.no_grad()
    def sample_anchor_error(self, timestep, like: torch.Tensor) -> Optional[torch.Tensor]:
        """Terminal error block for the anchor ('y') injection, drawn from the dedicated
        ``anchor_error_buffer`` (harvested from the predict step's last k frames).
        ``like`` is the pinned anchor block ([B,C,k,H,W]); the buffer's shape check makes a
        draw whose k differs simply miss, so changing k mid-run degrades to no injection
        rather than injecting a mis-shaped error.

        Probability gating is handled together with the noise/latent gates in
        ``draw_injection_flags`` so ``clean_prob`` can force the entire prediction
        draw -- including its anchor -- to stay clean.
        """
        if self.anchor_sample_from_all_grids:
            return self._sample_error_from_any_grid(self.anchor_error_buffer, like, "anchor")
        return self._sample_error_from_buffer(self.anchor_error_buffer, timestep, like, "anchor")

    def draw_injection_flags(self) -> Dict[str, bool]:
        """Per-draw injection decision for the GENERATION (predict) objective
        (SVI train_svi.py:1090-1111): independent noise/latent/y gates, overridden
        to all-clean with prob `clean_prob`."""
        if not self.inject_active:
            return {"noise": False, "latent": False, "y": False}
        flags = {
            "noise": random.random() < self.noise_prob,
            "latent": random.random() < self.latent_prob,
            "y": random.random() < self.y_prob,
        }
        if random.random() < self.clean_prob:
            flags = {"noise": False, "latent": False, "y": False}
        return flags

    def draw_memorize_injection(self) -> bool:
        """Per-draw injection decision for the MEMORIZE (inner/TTT) objective:
        corrupt the memorized chunk itself with prob `latent_prob`, overridden to
        clean with prob `clean_prob`. Unlike the generation gates, this corruption
        is CONSISTENT (input and target), replicating the test-time TTT update,
        which builds its target from the generated -- corrupted -- chunk."""
        if not self.inject_active:
            return False
        if random.random() >= self.latent_prob:
            return False
        if random.random() < self.clean_prob:
            return False
        return True

    def stats(self) -> Dict[str, float]:
        out = {
            "err_buffer/noise_nonempty_grids": float(sum(1 for v in self.noise_error_buffer.values() if v)),
            "err_buffer/y_nonempty_grids": float(sum(1 for v in self.y_error_buffer.values() if v)),
            "err_buffer/anchor_nonempty_grids": float(sum(1 for v in self.anchor_error_buffer.values() if v)),
            "err_buffer/anchor_samples": float(sum(len(v) for v in self.anchor_error_buffer.values())),
            "err_buffer/total_samples": float(
                sum(len(v) for v in self.noise_error_buffer.values())
                + sum(len(v) for v in self.y_error_buffer.values())
                + sum(len(v) for v in self.anchor_error_buffer.values())
            ),
            "err_buffer/num_harvested": float(self.num_harvested),
            "err_buffer/num_injected": float(sum(self._inj_ok.values())),
            "err_buffer/inject_active": float(self.inject_active),
        }
        for t in ("noise", "y", "anchor"):
            # #1 mean harvested error scale (per-element RMS). Watch for upward drift
            #    (feedback) or collapse to 0 (converged / injection inert). Expect
            #    anchor >= y (terminal error > one-step error).
            out[f"err_buffer/harv_rms_{t}"] = self._harv_rms_sum[t] / max(1, self._harv_rms_cnt[t])
            # #2 injection miss rate: attempts that found no matching sample and injected
            #    nothing. High => anti-drift silently off for that buffer. 0 when no attempts.
            out[f"err_buffer/miss_rate_{t}"] = (
                1.0 - self._inj_ok[t] / self._inj_req[t] if self._inj_req[t] else 0.0
            )
            # #3 mean relative corruption strength ||err|| / ||x0|| over successful injects.
            out[f"err_buffer/inject_rel_{t}"] = self._inj_rel_sum[t] / max(1, self._inj_ok[t])
        total_req = sum(self._inj_req.values())
        out["err_buffer/miss_rate"] = (
            1.0 - sum(self._inj_ok.values()) / total_req if total_req else 0.0
        )
        # #4 harvest source split -- confirms the predict-side harvest actually fires
        #    (if ~0, the anchor buffer and compounding errors never fill).
        out["err_buffer/num_harvested_memorize"] = float(self.num_harvested_memorize)
        out["err_buffer/num_harvested_predict"] = float(self.num_harvested_predict)
        tot_h = self.num_harvested_memorize + self.num_harvested_predict
        out["err_buffer/predict_harvest_frac"] = self.num_harvested_predict / tot_h if tot_h else 0.0
        return out


@contextlib.contextmanager
def _override_checkpointed_blocks(dit, params_override):
    """Run each DiT block under activation checkpointing while keeping the LoRA
    override active *through* the recompute.

    A plain ``_reparametrize`` context (used on the non-checkpointed path) swaps the
    LoRA params in for the duration of the forward and restores phi_0 on exit -- i.e.
    *before* the backward. Non-reentrant activation checkpointing recomputes the
    forward during backward, so it would see the restored phi_0 and silently corrupt
    the second-order meta-gradient. That is why the original code force-disabled
    gradient checkpointing whenever an override was present.

    To make them compatible, we checkpoint each block individually and pass that
    block's override tensors as *explicit checkpoint inputs*. ``torch.utils.checkpoint``
    saves those inputs and feeds them back into the recomputed closure, where we
    re-apply them with ``_reparametrize`` on the block. The override is therefore live
    during both the initial forward and every recompute (inner ``create_graph`` grad
    *and* the outer meta-backward), so the meta-gradient matches the non-checkpointed
    path -- while only each block's output (not its internal activations) is retained.
    """
    blocks = list(dit.blocks)
    originals = [b.forward for b in blocks]

    def _make_patched(block, orig_forward, rel_names, ovr_tensors):
        def patched(x, context, t_mod, freqs):
            def run(x_, context_, t_mod_, freqs_, *ovr):
                params = {n: t for n, t in zip(rel_names, ovr)}
                # math SDPA must also be pinned inside the recompute, which runs
                # during backward outside the outer _maybe_math_sdpa scope.
                with _maybe_math_sdpa(), _reparametrize(block, params, tie_weights=False, strict=False):
                    return orig_forward(x_, context_, t_mod_, freqs_)
            # Disable early-stop recomputation: it raises an internal
            # _StopRecomputationError *into* the context managers wrapping the
            # recompute (the _reparametrize / _maybe_math_sdpa generators), which
            # then fail with "generator didn't stop after throw()". Early stop is
            # only a perf optimisation, so turning it off recomputes the full block
            # and unwinds the CMs normally.
            with torch.utils.checkpoint.set_checkpoint_early_stop(False):
                return torch.utils.checkpoint.checkpoint(
                    run, x, context, t_mod, freqs, *ovr_tensors, use_reentrant=False,
                )
        return patched

    try:
        for i, block in enumerate(blocks):
            prefix = f"blocks.{i}."
            rel_names, ovr_tensors = [], []
            for full_name, t in params_override.items():
                if full_name.startswith(prefix):
                    rel_names.append(full_name[len(prefix):])
                    ovr_tensors.append(t)
            block.forward = _make_patched(block, originals[i], rel_names, ovr_tensors)
        yield
    finally:
        for block, orig in zip(blocks, originals):
            block.__dict__.pop("forward", None)  # restore the class-method dispatch


def _model_fn_with_override(pipe, dit, params_override, *, use_gradient_checkpointing=False, **model_fn_kwargs):
    """Call ``pipe.model_fn`` (the Wan DiT forward) optionally with the DiT's LoRA
    parameters replaced by ``params_override`` so that autograd flows through the
    override tensors (the key to second-order meta-learning).

    With ``use_gradient_checkpointing`` the override is carried through the per-block
    checkpoint recompute (see ``_override_checkpointed_blocks``); without it we fall
    back to a plain whole-model ``_reparametrize`` and retain all activations."""
    if params_override is None:
        return pipe.model_fn(dit=dit, use_gradient_checkpointing=use_gradient_checkpointing, **model_fn_kwargs)
    if use_gradient_checkpointing:
        # We do our own per-block checkpointing, so model_fn must not double-wrap.
        with _override_checkpointed_blocks(dit, params_override):
            return pipe.model_fn(dit=dit, use_gradient_checkpointing=False, **model_fn_kwargs)
    with _maybe_math_sdpa(), _reparametrize(dit, params_override, tie_weights=False, strict=False):
        return pipe.model_fn(dit=dit, **model_fn_kwargs)


def compute_flow_matching_loss(
    pipe,
    scheduler: FlowMatchScheduler,
    x0: torch.Tensor,            # clean clip latents [B, C, T', H', W']
    context: torch.Tensor,       # text embedding [B, L, D]
    inner_cfg: InnerLoopConfig,
    *,
    params_override: Optional[Dict[str, torch.Tensor]] = None,
    use_gradient_checkpointing: bool = False,
    condition_on_first_frame: bool = False,
    sink_frame: Optional[torch.Tensor] = None,
    num_anchor_latent_frames: int = 1,
    anchor_latents: Optional[torch.Tensor] = None,
    dit=None,
    error_recycler: Optional[ErrorRecycler] = None,
    inject_memorize_error: bool = False,
    inject_generation_error: bool = False,
    inject_anchor_error: bool = False,
    harvest_errors: bool = False,
) -> torch.Tensor:
    """Rectified flow-matching loss matching DiffSynth's ``FlowMatchSFTLoss``.

    ``noise_pred = model_fn(z_t, t, c)`` and target ``= noise - x0``; averaged over
    ``num_mc_samples`` random (t, noise) draws.

    ``condition_on_first_frame`` replicates Wan2.2-TI2V-5B's fused first-frame (image)
    conditioning used at inference (``WanVideoUnit_ImageEmbedderFused`` +
    ``inputs["latents"][:, :, 0:1] = first_frame_latents`` each denoising step): the
    first latent frame is pinned to its clean value (never noised), ``model_fn`` is told
    ``fuse_vae_embedding_in_latents=True`` so the DiT's ``seperated_timestep`` path gives
    that frame timestep 0 (treated as clean context), and it is excluded from the loss
    (it is given, not predicted). Because the Wan causal VAE's first latent frame is the
    single-frame encoding of the clip's first pixel frame, pinning ``x0[:, :, 0:1]`` is
    exactly conditioning on that frame -- so when the caller passes an overlap-chunked
    clip whose first frame is the previous chunk's last frame, this matches the
    autoregressive anchor used by ``WanE2ETTTSequentialGenerator``. Only meaningful for a
    DiT with ``fuse_vae_embedding_in_latents`` (TI2V-5B); no-op for clips with <2 latent
    frames.

    ``sink_frame`` (E2E-TTT first-frame "sink", optional, only meaningful together with
    ``condition_on_first_frame``): a second clean anchor frame -- the video's very first
    latent frame -- pinned *before* the usual local anchor, so the model conditions on
    ``[sink_frame, x0[:, :, 0:1]]`` (2 clean leading frames) instead of just
    ``x0[:, :, 0:1]``. This gives every predicted chunk a fixed, non-sliding reference to
    how the video started, on top of the sliding previous-chunk anchor.

    ``num_anchor_latent_frames`` (k) widens the LOCAL anchor from 1 latent frame to a
    contiguous block of k. A single anchor frame is motion-ambiguous -- no velocity can be
    read off one frame -- so with k=1 the only channel carrying "which way was it moving" is
    the LoRA scratchpad, whose (unconditioned) memorize objective is dominated by appearance
    rather than direction. k>1 puts the velocity directly in the conditioning.

    Frame layout, all slices taken from the caller's overlap-chunked ``x0`` so the anchor
    latents are genuine positional latents of a contiguous encode (never standalone
    single-frame encodes dropped at position >0, whose VAE statistics would be wrong):

      * no sink:   ``x0[:, :, 0:k]``      at latent positions ``0..k-1``   (num_clean = k)
      * with sink: ``x0[:, :, 1:1+k]``    at latent positions ``1..k``     (num_clean = k+1)

    With a sink the block is displaced by one position, so ``x0``'s own latent 0 is unused
    (displaced from the input by the sink, excluded from the loss by ``num_clean``) and
    serves only as the VAE causal context that makes latents ``1..k`` correct. The caller
    must slice chunks with ``anchor_overlap_pixel_frames(k, use_sink)`` of overlap for these
    slices to actually be the preceding chunk's tail.

    ``anchor_latents`` is the ALTERNATIVE conditioning route (see ``attention_anchor_latents``)
    and is mutually exclusive with everything above. Instead of overwriting ``x0``'s leading
    latents, the anchor block is prepended to the DiT's token sequence as extra clean
    positions, so ``x0`` is noised and supervised in full -- no frames are pinned, none are
    dropped from the loss, and the chunk keeps every latent it has. That is what makes this
    route usable on a plain T2V DiT with no fused first-frame conditioning at all, and why the
    caller chunks contiguously rather than with an overlap.
    """
    if anchor_latents is not None and (condition_on_first_frame or sink_frame is not None):
        raise ValueError(
            "anchor_latents (attention anchor) and condition_on_first_frame/sink_frame (fused "
            "anchor) are two different conditioning mechanisms; pass exactly one."
        )
    if x0.dim() != 5:
        raise ValueError(f"Expected x0 [B,C,T,H,W], got {tuple(x0.shape)}")
    dit = dit if dit is not None else pipe.dit
    device, dtype = pipe.device, pipe.torch_dtype
    num_ts = len(scheduler.timesteps)
    min_ts = int(inner_cfg.min_timestep_boundary * num_ts)
    max_ts = max(min_ts + 1, int(inner_cfg.max_timestep_boundary * num_ts))

    # First-frame conditioning is only the TI2V-5B fused path. num_clean is the number of
    # pinned leading frames: the k-frame local anchor block plus 1 when a sink is also given.
    # It needs at least one latent frame to predict after the pinned anchor(s); otherwise fall
    # back to the plain unconditioned objective.
    sink_frame = sink_frame if condition_on_first_frame else None
    k_anchor = max(1, int(num_anchor_latent_frames))
    num_clean = num_clean_latents(k_anchor, sink_frame is not None)
    # On the contiguous (k>1) path a sink displaces the anchor block to latent positions
    # 1..k, so it is sliced from x0 at offset 1; x0's own latent 0 is then unused except as
    # the VAE causal context that makes latents 1..k positionally correct.
    #
    # k == 1 must stay at offset 0. There the legacy layout applies: the chunk overlap is a
    # single frame (anchor_overlap_pixel_frames returns 1), so x0's latent 0 -- not latent 1
    # -- is the previous chunk's last frame, and the sink is *prepended* to it rather than
    # overwriting a contiguous encode. Shifting this case would silently anchor every chunk
    # on the wrong frame and invalidate phi_0 checkpoints trained before k existed.
    anchor_start = 1 if (sink_frame is not None and k_anchor > 1) else 0
    condition_on_first_frame = bool(condition_on_first_frame) and x0.shape[2] >= num_clean + 1

    # With a differentiable params_override, checkpointing is only safe because
    # _model_fn_with_override carries the override through the per-block recompute
    # (see _override_checkpointed_blocks). A plain whole-model _reparametrize would
    # restore phi_0 before backward and corrupt the meta-gradient, so the override
    # checkpointing path threads phi_k in as explicit checkpoint inputs instead.
    gc = use_gradient_checkpointing

    total = torch.zeros((), device=device)
    for _ in range(max(1, int(inner_cfg.num_mc_samples))):
        timestep_id = torch.randint(min_ts, max_ts, (1,))
        timestep = scheduler.timesteps[timestep_id].to(dtype=dtype, device=device)
        noise = torch.randn_like(x0)
        # --- Error recycling. All injected tensors are detached buffer samples --
        # pure data perturbation, invisible to the (second-order) meta-graph.
        # Two distinct injection modes:
        #
        # (1) inject_memorize_error (inner/TTT objective): corrupt the memorized
        #     chunk itself, CONSISTENTLY (input and target). At test time
        #     ttt_update_inplace builds its target from the generated -- corrupted --
        #     chunk; there is no clean x0 there, so the meta-trained inner update
        #     must see the same fully-corrupted loss it will run at inference.
        #     Fresh clean Gaussian noise, exactly like the test-time update.
        #
        # (2) inject_generation_error (predict/meta objective): SVI's asymmetric
        #     scheme (train_svi.py:1114-1139) -- corrupt the noise and/or the
        #     add_noise input while the target keeps pointing at the CLEAN chunk,
        #     so the learned velocity corrects mid-trajectory sampling errors.
        injected_this_draw = False
        # Tracked apart from injected_this_draw because it alone contaminates the harvest:
        # noise injection puts the recycled error into the TARGET (target = noise_w_error -
        # x0), so re-harvesting would algebraically re-deposit it. Latent/anchor injection
        # corrupts only the INPUT, so its residual is a genuine corrupted-input (compounding)
        # error -- safe, and exactly the train-test-matching signal we want to collect.
        noise_injected_this_draw = False
        injection_flags = {"noise": False, "latent": False, "y": False}
        if error_recycler is not None and (inject_generation_error or inject_anchor_error):
            # Draw all prediction-side gates together so a single clean_prob
            # override also keeps the conditioning anchor clean.
            injection_flags = error_recycler.draw_injection_flags()
        x0_used = x0
        if error_recycler is not None and inject_memorize_error and error_recycler.draw_memorize_injection():
            err = error_recycler.sample_latent_error(timestep, like=x0)
            if err is not None:
                x0_used = x0 + err  # the chunk as it would arrive at test time
                injected_this_draw = True
        noise_w_error, latents_w_error = noise, x0_used
        if error_recycler is not None and inject_generation_error:
            if injection_flags["noise"]:
                err = error_recycler.sample_noise_error(timestep, like=x0)
                if err is not None:
                    noise_w_error = noise + err
                    injected_this_draw = True
                    noise_injected_this_draw = True
            if injection_flags["latent"]:
                err = error_recycler.sample_latent_error(timestep, like=x0)
                if err is not None:
                    latents_w_error = latents_w_error + err
                    injected_this_draw = True
        latents = scheduler.add_noise(latents_w_error, noise_w_error, timestep)
        # Memorize: target follows x0_used (corrupted chunk -> corrupted target,
        # test-time-faithful). Predict: x0_used == x0, so the target stays clean
        # even under generation-error injection (SVI's self-correcting supervision).
        target = scheduler.training_target(x0_used, noise_w_error, timestep)
        if condition_on_first_frame:
            # Pin the leading latent frame(s) to the clean anchor(s) (no noise), exactly as
            # the inference pipeline re-clamps latents[:, :, 0:num_clean] = first_frame_latents.
            # Only the LOCAL anchor (the last of the num_clean slots) gets SVI's anchor-error
            # injection -- it is the frame that actually drifts at inference (a generated
            # frame carried forward); the sink frame is a fixed reference captured once, so
            # it is left clean.
            local_anchor = x0[:, :, anchor_start:anchor_start + k_anchor]
            if error_recycler is not None and inject_anchor_error and injection_flags["y"]:
                # SVI's y-error injection (train_svi.py:1118-1130): corrupt the
                # conditioning frame(s) with a recycled terminal error, simulating the
                # drifted previous-chunk frames the anchor actually is at inference. Drawn
                # from the dedicated anchor buffer (last-frame terminal errors), not the
                # mid-trajectory full-chunk y buffer. Supervision stays clean.
                #
                # For k>1 this MUST be one contiguous k-frame block, not k independent
                # single-frame draws: independent draws are temporally incoherent and would
                # corrupt the very velocity cue the wider anchor exists to provide. The
                # buffer stores k-frame blocks harvested from the predict chunk's last k
                # latents (exactly the frames that become the next anchor), so a single
                # shape-matched draw is already block-coherent.
                a_err = error_recycler.sample_anchor_error(timestep, like=local_anchor)
                if a_err is not None:
                    local_anchor = local_anchor + a_err
            anchor = torch.cat([sink_frame, local_anchor], dim=2) if sink_frame is not None else local_anchor
            latents = torch.cat([anchor, latents[:, :, num_clean:]], dim=2)
        anchor_in = anchor_latents
        if anchor_in is not None and error_recycler is not None and inject_anchor_error and injection_flags["y"]:
            # Same SVI y-error injection as the fused path, applied to the prefix block: at
            # inference this block is the previous chunk's *generated* (hence drifted) tail,
            # so phi_0 must be meta-learned under that corruption. Only the trailing k frames
            # are injected into -- when a sink leads the block it is a fixed reference frame
            # captured once, which never drifts. Supervision stays clean either way, since the
            # prefix carries no target.
            k_inject = min(k_anchor, anchor_in.shape[2])
            a_err = error_recycler.sample_anchor_error(timestep, like=anchor_in[:, :, -k_inject:])
            if a_err is not None:
                anchor_in = torch.cat([anchor_in[:, :, :-k_inject], anchor_in[:, :, -k_inject:] + a_err], dim=2)
        noise_pred = _model_fn_with_override(
            pipe, dit, params_override,
            latents=latents, timestep=timestep, context=context,
            use_gradient_checkpointing=gc,
            fuse_vae_embedding_in_latents=condition_on_first_frame,
            num_fused_clean_frames=num_clean,
            anchor_latents=anchor_in,
        )
        if error_recycler is not None and harvest_errors:
            # Recycle this draw's own prediction error into the buffers (SVI
            # train_svi.py:1151-1160). Free: no extra forward pass. What is safe to
            # harvest differs by objective:
            if inject_memorize_error and not injected_this_draw:
                # MEMORIZE: clean draws only. Consistent corruption makes the target
                # (noise - x0_used) algebraically re-deposit the injected error, and the
                # inner loss that would damp it acts on discarded adapted weights under a
                # non-meta objective -- so injected memorize draws would contaminate the
                # buffers. Unconditioned (no anchor frame in the memorize objective).
                error_recycler.harvest(scheduler, noise_pred, target, latents, timestep, source="memorize")
            elif inject_generation_error and not noise_injected_this_draw:
                # PREDICT: harvest genuine corrupted-input (compounding) errors -- the
                # drifted-input regime the buffers otherwise never see. The asymmetric
                # target is built from CLEAN x0, so latent/anchor injection (input-only)
                # leaves no algebraic residue; only NOISE injection leaks into the target,
                # so those draws are skipped. condition_on_first_frame routes the last
                # frame's terminal error to the anchor buffer and drops the pinned leading
                # frame(s).
                error_recycler.harvest(
                    scheduler, noise_pred, target, latents, timestep,
                    condition_on_first_frame=condition_on_first_frame, num_clean_frames=num_clean,
                    num_anchor_frames=k_anchor, attention_anchor=anchor_latents is not None,
                    source="predict",
                )
        if condition_on_first_frame:
            # The anchor frame(s) are given (timestep 0), not predicted -- supervise only
            # the continuation frames, matching what the model actually generates at inference.
            loss = F.mse_loss(noise_pred[:, :, num_clean:].float(), target[:, :, num_clean:].float())
        else:
            loss = F.mse_loss(noise_pred.float(), target.float())
        total = total + loss * scheduler.training_weight(timestep)
    return total / max(1, int(inner_cfg.num_mc_samples))


# --------------------------------------------------------------------------- #
# Meta-training inner loop (memorize->predict for MAML/FOMAML; SGD for Reptile)#
# --------------------------------------------------------------------------- #


def _inner_adapt_stats(base_params, final_lora, lora_names, device):
    """Inner adaptation magnitude for the last task: how far the inner loop moved the
    LoRA scratchpad from phi_0 (||phi_adapted - phi_0|| and the same / ||phi_0||).
    Fully detached -- a pure diagnostic, never part of the meta-graph."""
    out = {}
    if final_lora is None:
        return out
    with torch.no_grad():
        delta_sq = torch.zeros((), device=device)
        phi0_sq = torch.zeros((), device=device)
        for n in lora_names:
            phi0 = base_params[n].detach()
            delta_sq = delta_sq + (final_lora[n].detach() - phi0).pow(2).sum()
            phi0_sq = phi0_sq + phi0.pow(2).sum()
        inner_delta = delta_sq.sqrt()
        out["inner_adapt_norm"] = inner_delta
        out["inner_adapt_ratio"] = inner_delta / (phi0_sq.sqrt() + 1e-8)
    return out


def _run_reptile_inner_loop(
    pipe,
    scheduler: FlowMatchScheduler,
    video_chunks: List[List[torch.Tensor]],
    video_contexts: List[torch.Tensor],
    inner_cfg: InnerLoopConfig,
    *,
    learned_lrs: Optional[MetaLearnedLRSchedule] = None,
    use_gradient_checkpointing: bool = False,
    write_back: bool = False,
    error_recycler: Optional[ErrorRecycler] = None,
):
    """Reptile meta-update.

    Adapt phi_0 -> phi_K with plain (non-differentiable) SGD on the *same* memorize
    chunks MAML uses (chunks 0..N-2), then move phi_0 toward the adapted weights. The
    Reptile pseudo-gradient ``g = phi_0 - phi_K`` (averaged over the videos in the
    batch) is deposited onto the real phi_0 LoRA leaves through a surrogate scalar
    whose gradient w.r.t. phi_0 equals ``g``::

        surrogate = sum_n < phi_0[n], (phi_0[n] - phi_K[n]).detach() >
        d surrogate / d phi_0[n] = (phi_0[n] - phi_K[n]) = g[n]

    So the unmodified outer loop (``meta_loss.backward()`` -> outer optimizer step)
    applies the Reptile update with no special-casing in the runner. There is no
    predict term and no second-order graph: every inner step is a single first-order
    backward, so fused attention is fine (no double-backward requirement)."""
    dit = pipe.dit
    base_params = dict(dit.named_parameters())
    lora_names = [n for n in base_params if "lora" in n and base_params[n].requires_grad]
    if not lora_names:
        raise ValueError("No trainable LoRA parameters found on pipe.dit. Inject LoRA first.")

    device = pipe.device
    opt = make_inner_optimizer(inner_cfg)

    mem_loss_sum = torch.zeros((), device=device)
    mem_count = 0
    mem_step_idx = 0
    num_tasks = 0
    pseudo = {n: torch.zeros_like(base_params[n]) for n in lora_names}
    final_lora: Optional[Dict[str, torch.Tensor]] = None

    def _resolve_lrs(step_idx: int):
        return None if learned_lrs is None else learned_lrs.get_lrs(step_idx)

    for chunks, ctx in zip(video_chunks, video_contexts):
        if len(chunks) < 2:
            continue  # match MAML/FOMAML: need >=2 chunks to define the memorize set
        # Detached scratchpad initialised at phi_0; a leaf so autograd.grad can target it.
        current = {n: base_params[n].detach().clone().requires_grad_(True) for n in lora_names}
        for k in range(len(chunks) - 1):  # memorize chunks 0..N-2 (MAML's inner-loop set)
            for _ in range(max(1, int(inner_cfg.num_gradient_steps))):
                loss_mem = compute_flow_matching_loss(
                    pipe, scheduler, chunks[k], ctx, inner_cfg,
                    params_override={**current},
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    error_recycler=error_recycler,
                    inject_memorize_error=True, harvest_errors=True,
                )
                mem_loss_sum = mem_loss_sum + loss_mem.detach()
                mem_count += 1
                # First-order grad only (no create_graph): ordinary SGD adaptation.
                grads_list = torch.autograd.grad(
                    loss_mem, [current[n] for n in lora_names], allow_unused=True,
                )
                grads = {n: g for n, g in zip(lora_names, grads_list)}
                updated = opt.step(current, grads, learned_lrs=_resolve_lrs(mem_step_idx))
                # Detach + re-leaf so the next step differentiates w.r.t. the new weights only.
                current = {n: updated[n].detach().requires_grad_(True) for n in lora_names}
                mem_step_idx += 1
        with torch.no_grad():
            for n in lora_names:
                pseudo[n] = pseudo[n] + (base_params[n].detach() - current[n].detach())
        final_lora = current
        num_tasks += 1

    if num_tasks == 0:
        raise ValueError("No Reptile tasks produced; need >=2 chunks per video. Check chunking config.")

    if write_back and final_lora is not None:
        with torch.no_grad():
            for n, v in final_lora.items():
                base_params[n].copy_(v.detach())

    # Surrogate whose grad w.r.t. phi_0 equals the mean Reptile pseudo-gradient (phi_0-phi_K).
    meta_loss = torch.zeros((), device=device)
    for n in lora_names:
        meta_loss = meta_loss + (base_params[n].float() * (pseudo[n] / num_tasks).float()).sum()

    mem_mean = (mem_loss_sum / mem_count) if mem_count else mem_loss_sum
    stats = {
        "memorize_loss": mem_mean,
        "monitor_loss": mem_mean,   # no predict term in Reptile; track the memorize fit
        "num_pred": 0,
        "num_mem_steps": mem_count,
    }
    stats.update(_inner_adapt_stats(base_params, final_lora, lora_names, device))
    return meta_loss, stats


def run_meta_inner_loop(
    pipe,
    scheduler: FlowMatchScheduler,
    video_chunks: List[List[torch.Tensor]],   # per video: ordered chunks [1,C,T',H',W']
    video_contexts: List[torch.Tensor],        # per video: single narrative context [1,L,D]
    inner_cfg: InnerLoopConfig,
    *,
    truncate_steps: Optional[List[int]] = None,
    learned_lrs: Optional[MetaLearnedLRSchedule] = None,
    use_gradient_checkpointing: bool = False,
    write_back: bool = False,
    algorithm: Optional[str] = None,
    condition_on_first_frame: bool = False,
    condition_on_first_frame_sink: bool = True,
    num_anchor_latent_frames: int = 1,
    attention_anchor: bool = False,
    error_recycler: Optional[ErrorRecycler] = None,
):
    """Memorize->predict inner loop over chunk sequences (MAML / FOMAML / Reptile).

    For each video, starting from the LoRA meta-init phi_0 (a fresh clone of the real
    parameters, keeping the graph connected to phi_0):

        for k in 0 .. N-2:
            memorize chunk k   -> inner SGD step(s) on LoRA   (phi_k -> phi_{k+1})
            predict  chunk k+1 -> accumulate L'_{k+1}(phi_{k+1}) into the meta-loss

    Returns ``(meta_loss, stats)``. ``meta_loss.backward()`` populates grads on the
    real LoRA leaves (phi_0); with ``write_back=False`` the real leaves are never
    mutated, so the outer optimizer updates phi_0 directly. By default the meta-loss
    is second-order (create_graph=True). With FOMAML the inner grads are detached,
    dropping the Hessian term and the double-backward requirement. ``algorithm`` selects
    the variant (defaults to ``inner_cfg.algorithm``); ``algorithm="reptile"`` dispatches
    to the SGD-then-interpolate Reptile path (no predict term, no second-order graph).

    ``condition_on_first_frame`` (TI2V-5B only) conditions the *predict* loss on each
    chunk's first latent frame, replicating the fused first-frame anchor the inference
    generator pins to the previous chunk's last frame. It assumes overlap-chunked input
    (consecutive chunks share their boundary frame); the memorize objective stays
    unconditioned, matching the test-time ``ttt_update_inplace``.

    ``condition_on_first_frame_sink`` (on by default, requires ``condition_on_first_frame``)
    additionally pins each video's very first chunk's first latent frame -- a fixed,
    non-sliding "sink" -- alongside the sliding local anchor, for every predict step of that
    video. Like ``condition_on_first_frame``, it never applies to the memorize objective.

    ``attention_anchor`` selects the other conditioning route instead (Wan2.1-T2V and any DiT
    without fused first-frame conditioning): the predict loss for chunk k+1 is conditioned on
    a prefix of clean tokens built from chunk k's trailing latents (plus the sink), so chunk
    k+1 itself stays fully noised and fully supervised. It expects CONTIGUOUS, NON-OVERLAPPING
    chunks -- the anchor is read from the preceding chunk directly, not from an overlap -- and
    is mutually exclusive with ``condition_on_first_frame``.
    """
    if attention_anchor and condition_on_first_frame:
        raise ValueError(
            "attention_anchor and condition_on_first_frame are alternative conditioning "
            "mechanisms (prefix tokens vs fused pinned latents); enable exactly one."
        )
    condition_on_first_frame_sink = bool(condition_on_first_frame_sink) and (
        condition_on_first_frame or attention_anchor
    )
    algorithm = (algorithm or getattr(inner_cfg, "algorithm", "maml")).lower()
    if error_recycler is not None:
        # One outer step per call (batch of one video per rank); drives the warmup gate.
        error_recycler.begin_iteration()
    if algorithm == "reptile":
        # Reptile has no predict term -- it only ever runs the (unconditioned) memorize
        # objective -- so first-frame conditioning does not apply and is ignored here.
        return _run_reptile_inner_loop(
            pipe, scheduler, video_chunks, video_contexts, inner_cfg,
            learned_lrs=learned_lrs,
            use_gradient_checkpointing=use_gradient_checkpointing,
            write_back=write_back,
            error_recycler=error_recycler,
        )
    first_order = algorithm != "maml"  # FOMAML drops the Hessian
    truncate_steps = truncate_steps or []
    if first_order and truncate_steps:
        # FOMAML keeps phi connected to phi_0 only through the clone+update identity
        # path; detaching at a truncation step would sever that path and zero the
        # meta-gradient on phi_0. Truncation is meaningless without a second-order
        # graph, so ignore it.
        truncate_steps = []
    dit = pipe.dit
    base_params = dict(dit.named_parameters())
    lora_names = [n for n in base_params if "lora" in n and base_params[n].requires_grad]
    if not lora_names:
        raise ValueError("No trainable LoRA parameters found on pipe.dit. Inject LoRA first.")

    device = pipe.device
    opt = make_inner_optimizer(inner_cfg)

    meta_loss = torch.zeros((), device=device)
    mem_loss_sum = torch.zeros((), device=device)
    num_pred = 0
    mem_count = 0
    mem_step_idx = 0

    def _resolve_lrs(step_idx: int):
        if learned_lrs is None:
            return None
        return learned_lrs.get_lrs(step_idx)

    final_lora: Optional[Dict[str, torch.Tensor]] = None

    for chunks, ctx in zip(video_chunks, video_contexts):
        if len(chunks) < 2:
            continue  # need >=2 chunks to form a memorize->predict pair

        # Fresh LoRA init phi_0 for this video; clone keeps the graph to the real leaves.
        current_lora: Dict[str, torch.Tensor] = {n: base_params[n].clone() for n in lora_names}

        # Fixed global anchor for this video: the very first chunk's first latent frame,
        # pinned alongside the sliding local anchor in every predict step below. Always
        # latent 0 of chunk 0 -- a position-0 latent used at position 0, so its VAE
        # statistics are correct regardless of how wide the local anchor block is.
        sink_frame = chunks[0][:, :, 0:1] if condition_on_first_frame_sink else None

        for k in range(len(chunks) - 1):
            # ---- memorize chunk k ----
            for _ in range(max(1, int(inner_cfg.num_gradient_steps))):
                params_override = {**current_lora}
                # Memorize with recycled errors corrupting the chunk CONSISTENTLY
                # (input and target), replicating the test-time TTT update, which
                # builds its target from the generated -- corrupted -- chunk. The
                # anti-drift signal comes from the outer loop: phi_0 is meta-shaped
                # so that memorizing corrupted chunks still predicts clean ones.
                # Clean draws' residuals are harvested back into the buffers.
                loss_mem = compute_flow_matching_loss(
                    pipe, scheduler, chunks[k], ctx, inner_cfg,
                    params_override=params_override,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    error_recycler=error_recycler,
                    inject_memorize_error=True, harvest_errors=True,
                )
                mem_loss_sum = mem_loss_sum + loss_mem.detach()
                mem_count += 1
                # Second-order: create_graph=True keeps grads differentiable so the
                # outer backward carries the Hessian term. First-order (FOMAML):
                # create_graph=False detaches grads (and frees the inner-loss graph,
                # since retain_graph defaults to create_graph), leaving phi connected
                # to phi_0 only via the clone+update identity path.
                # TODO: add checkpoint gradient here.
                grads_list = torch.autograd.grad(
                    loss_mem, [current_lora[n] for n in lora_names],
                    create_graph=not first_order, allow_unused=True,
                )
                grads = {n: g for n, g in zip(lora_names, grads_list)}
                current_lora = opt.step(current_lora, grads, learned_lrs=_resolve_lrs(mem_step_idx))
                if mem_step_idx in truncate_steps:
                    # Truncated BPTT: cut the meta-gradient through this step.
                    current_lora = {kk: vv.detach().requires_grad_(True) for kk, vv in current_lora.items()}
                mem_step_idx += 1

            # ---- predict next chunk k+1 (meta term) ----
            # At inference each follow-up chunk is generated with TI2V-5B's fused
            # first-frame conditioning on the previous chunk's last frame; with overlap
            # chunking that anchor IS chunks[k+1]'s first frame, so condition the predict
            # loss on it to match the inference-time objective (see compute_flow_matching_loss).
            # The predict (meta) loss carries SVI's generation-side error recycling:
            # noise/latent errors corrupt the inputs while the TARGET stays clean
            # (predict defines what "good" means), teaching the adapted model to
            # self-correct mid-trajectory sampling errors. The pinned anchor frame
            # -- a *generated* frame at inference -- additionally gets SVI's y-error
            # injection, so phi_0 is meta-learned under realistic anchor corruption.
            # Attention-anchor route: the anchor is a prefix built from chunk k's trailing
            # latents (the frames inference will actually hand forward) plus the fixed sink,
            # and chunk k+1 is left untouched -- fully noised, fully supervised.
            anchor_prefix = (
                attention_anchor_latents(chunks[k], num_anchor_latent_frames, sink_frame)
                if attention_anchor else None
            )
            loss_pred = compute_flow_matching_loss(
                pipe, scheduler, chunks[k + 1], ctx, inner_cfg,
                params_override={**current_lora},
                use_gradient_checkpointing=use_gradient_checkpointing,
                condition_on_first_frame=condition_on_first_frame,
                sink_frame=None if attention_anchor else sink_frame,
                num_anchor_latent_frames=num_anchor_latent_frames,
                anchor_latents=anchor_prefix,
                error_recycler=error_recycler,
                inject_generation_error=True,
                inject_anchor_error=condition_on_first_frame or attention_anchor,
                # Harvest the predict step's genuine corrupted-input (compounding) errors:
                # full-chunk noise/data errors into the shared buffers and the last frame's
                # terminal error into the anchor buffer. Noise-injected draws are skipped
                # inside compute_flow_matching_loss (their error leaks into the target).
                harvest_errors=True,
            )
            meta_loss = meta_loss + loss_pred
            num_pred += 1

        final_lora = current_lora

    if num_pred == 0:
        raise ValueError("No memorize->predict pairs were produced; check chunking config.")

    if write_back and final_lora is not None:
        with torch.no_grad():
            for n, v in final_lora.items():
                base_params[n].copy_(v.detach())

    meta_loss = meta_loss / num_pred
    stats = {
        "memorize_loss": (mem_loss_sum / mem_count) if mem_count else mem_loss_sum,
        "monitor_loss": meta_loss.detach(),  # mean next-chunk predict loss
        "num_pred": num_pred,
        "num_mem_steps": mem_count,
    }
    stats.update(_inner_adapt_stats(base_params, final_lora, lora_names, device))
    return meta_loss, stats


# --------------------------------------------------------------------------- #
# Test-time first-order in-place TTT update                                   #
# --------------------------------------------------------------------------- #


class TestTimeInnerOptimizer:
    """Persistent test-time inner-loop optimizer for the LoRA scratchpad.

    Meta-training runs the inner loop *functionally*: ``make_inner_optimizer(inner_cfg)``
    is built once per outer step and its state (AdamW moments, Muon momentum) persists
    across the whole chunk sequence of a video, while phi is a graph-connected clone of
    phi_0. At test time phi_0 IS the live weight, so the update must be applied in place --
    but it has to remain the SAME update rule with the SAME state lifetime, or phi_0 is
    being evaluated under an adaptation procedure it was never meta-learned for.

    Two things this class exists to get right, both of which the previous hand-rolled
    in-place SGD got wrong:

      * **Optimizer parity.** ``inner_cfg.optimizer`` is honoured here exactly as in
        training. The old path hardcoded plain SGD and ignored the field entirely, so an
        AdamW-meta-trained phi_0 was adapted by a different rule with a per-step
        displacement orders of magnitude smaller: AdamW's preconditioned step moves each
        coordinate by ~lr, whereas SGD under ``_per_tensor_clip(g, 1.0)`` moves a whole
        tensor by at most ``lr`` in Frobenius norm, i.e. ~``lr/sqrt(numel)`` per
        coordinate. Analytically that is a factor ~``sqrt(numel)`` (~600x for a 128x3072
        LoRA factor); measured end-to-end on bf16 leaves at lr=1e-4/clip=1.0 over 2 steps
        it is ~7200x, the extra factor being the bf16 quantization discussed below.

      * **fp32 master weights.** The LoRA leaves are bf16 (8-bit mantissa), and at LoRA's
        parameter scale (~1e-2) the quantum is ~7e-5, well above a clipped SGD step's
        ~1.6e-7 per coordinate. An in-place bf16 ``sub_`` therefore discarded most of the
        update, and being in-place it could never accumulate what it discarded. The
        optimizer now runs on an fp32 master copy that is cast back into the live
        parameters after each step, so small updates accumulate exactly even when a single
        step is below the bf16 quantum. Note this is a *secondary* effect: measured on
        rank-128 factors with lr=1e-4/clip=1.0 over 2 steps, the master alone changes
        realized ||dphi||/||phi|| by only ~1.2x (2.0e-6 -> 2.4e-6, since the bf16 readback
        still quantizes each step), whereas switching sgd -> adamw changes it by ~7200x
        (2.0e-6 -> 1.5e-2). Meta-training keeps phi in bf16
        (``upcast_dtype=pipe.torch_dtype``), so fp32 here is a deliberate one-directional
        improvement; the live parameters the DiT reads stay bf16, so the forward is
        unchanged.

    State lifetime mirrors training: build one per video -- i.e. right after the scratchpad
    is reset to phi_0 -- and reuse it across that video's chunks.
    """

    def __init__(self, dit: nn.Module, inner_cfg: InnerLoopConfig):
        self.opt = make_inner_optimizer(inner_cfg)
        lora_params = get_trainable_lora_params(dit)
        if not lora_params:
            raise ValueError("No trainable LoRA params found for test-time TTT.")
        self.names = list(lora_params.keys())
        self.params = [lora_params[n] for n in self.names]
        # fp32 master copy of phi, seeded from the live (bf16) leaves.
        self.master: Dict[str, torch.Tensor] = {
            n: lora_params[n].detach().float().clone() for n in self.names
        }

    @torch.no_grad()
    def apply(self, grads: Sequence[Optional[torch.Tensor]]) -> None:
        """One optimizer step: advance the fp32 master from ``grads``, then write it back
        into the live bf16 parameters. Grads are upcast so the clipping and the moment
        updates are computed in fp32. ``max_inner_grad_norm`` is applied inside the
        optimizer (``_per_tensor_clip``), the same call the meta-trained inner loop makes,
        rather than re-implemented here."""
        grad_map = {
            n: (g.detach().float() if g is not None else None)
            for n, g in zip(self.names, grads)
        }
        self.master = self.opt.step(self.master, grad_map)
        for name, p in zip(self.names, self.params):
            p.copy_(self.master[name].to(dtype=p.dtype))


def ttt_update_inplace(
    pipe,
    scheduler: FlowMatchScheduler,
    x0: torch.Tensor,
    context: torch.Tensor,
    inner_cfg: InnerLoopConfig,
    *,
    use_gradient_checkpointing: bool = False,
    updater: Optional[TestTimeInnerOptimizer] = None,
) -> float:
    """``inner_cfg.num_gradient_steps`` first-order LoRA updates applied in place (no
    second-order graph). Used between chunks at test time. Returns the last loss value.

    ``updater`` carries the optimizer state (AdamW moments / Muon momentum) and the fp32
    master weights across calls, matching meta-training, where the inner optimizer is
    built once per outer step and persists across the video's whole chunk sequence.
    Passing ``None`` builds a throwaway one, which resets that state every call -- exact
    only for stateless SGD; a caller generating multi-chunk video should own one per video
    (see ``WanE2ETTTSequentialGenerator.generate``)."""
    if updater is None:
        updater = TestTimeInnerOptimizer(pipe.dit, inner_cfg)
    last = 0.0
    for _ in range(max(1, int(inner_cfg.num_gradient_steps))):
        with torch.enable_grad():
            loss = compute_flow_matching_loss(
                pipe, scheduler, x0, context, inner_cfg,
                params_override=None,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )
            grads = torch.autograd.grad(loss, updater.params, allow_unused=True)
        updater.apply(grads)
        last = float(loss.detach().item())
    return last


# --------------------------------------------------------------------------- #
# LoRA injection / loading for inference                                      #
# --------------------------------------------------------------------------- #


def inject_lora_for_ttt(
    pipe,
    lora_rank: int = 32,
    lora_alpha: Optional[int] = None,
    target_modules: str = "q,k,v,o,ffn.0,ffn.2",
    lora_checkpoint: Optional[str] = None,
):
    """Inject a PEFT LoRA adapter into ``pipe.dit`` and (optionally) load meta-trained
    phi_0 weights. Marks only LoRA params trainable so test-time TTT can update them.
    Returns the snapshot of phi_0."""
    from peft import LoraConfig, inject_adapter_in_model

    if lora_alpha is None:
        lora_alpha = lora_rank
    modules = target_modules.split(",") if isinstance(target_modules, str) else target_modules
    pipe.dit.requires_grad_(False)
    config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=modules)
    pipe.dit = inject_adapter_in_model(config, pipe.dit)

    # Keep LoRA params in the compute dtype and trainable.
    for n, p in pipe.dit.named_parameters():
        if "lora" in n:
            p.data = p.data.to(pipe.torch_dtype)
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    if lora_checkpoint is not None:
        _load_lora_checkpoint(pipe, lora_checkpoint)

    return snapshot_lora_state(pipe.dit)


def _load_lora_checkpoint(pipe, path: str):
    from ..core import load_state_dict

    state = load_state_dict(path)
    # The trainer saves with `remove_prefix_in_ckpt="pipe.dit."`, leaving keys relative
    # to the DiT, and may store either `lora_A.weight` or `lora_A.default.weight`.
    remapped = {}
    for k, v in state.items():
        if k.startswith("pipe.dit."):
            k = k[len("pipe.dit."):]
        if (".lora_A.weight" in k) or (".lora_B.weight" in k):
            k = k.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight")
        remapped[k] = v.to(pipe.torch_dtype)
    missing, unexpected = pipe.dit.load_state_dict(remapped, strict=False)
    loaded = len(remapped) - len(unexpected)
    print(f"[E2E-TTT] Loaded LoRA checkpoint {path}: {loaded} keys matched, "
          f"{len(unexpected)} unexpected.")


# --------------------------------------------------------------------------- #
# Test-time sequential generation                                             #
# --------------------------------------------------------------------------- #


class WanE2ETTTSequentialGenerator:
    """Sequential long-video generation with E2E-TTT for a Wan T2V/TI2V pipeline.

    For each narrative prompt: reset the LoRA scratchpad to phi_0, then for chunk k:
    generate chunk k -> *memorize* it with in-place first-order LoRA TTT -> generate
    chunk k+1 with the adapted LoRA. Chunks are concatenated into one long video.

    Subclasses customise via ``_chunk_prompts`` / ``_log_prefix``; the chunk loop itself
    (anchor handoff, sink, k>1 block, boundary trimming, per-video optimizer state) is not
    meant to be overridden -- a copy of it drifts out of sync with the anchoring rules.
    """

    _log_prefix = "[E2E-TTT]"

    def __init__(
        self,
        pipe,
        inner_cfg: InnerLoopConfig,
        infer_cfg: InferenceConfig,
        *,
        phi0: Optional[Dict[str, torch.Tensor]] = None,
        use_gradient_checkpointing: bool = False,
    ):
        # Off by default here; the Wan2.2-TI2V-5B drivers pass True (their
        # --no_gradient_checkpointing turns it off). At test time the memorize step runs
        # with params_override=None, so it takes the stock pipe.model_fn checkpointing --
        # numerically identical to retaining activations, at one extra forward per inner
        # step (2/chunk against 50 sampling forwards). Retaining them instead makes
        # inner-loop memory scale with frames_per_chunk with no headroom to spare at 720p:
        # a 704x1280 chunk OOMs on a 96GB H100 at frames_per_chunk=53 (14 latent frames)
        # while 49 (13) fits, i.e. the arm's chunk length silently decides whether TTT
        # runs at all.
        self.pipe = pipe
        self.inner_cfg = inner_cfg
        self.infer_cfg = infer_cfg
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.scheduler = make_training_scheduler(infer_cfg.sigma_shift)
        self.phi0 = phi0 if phi0 is not None else snapshot_lora_state(pipe.dit)
        # Echo the update rule phi_0 is about to be adapted with: it must match the
        # meta-training run's --e2e_inner_optimizer / --e2e_inner_lr / --e2e_num_gradient_steps
        # (note the effective step count is ttt_steps_per_chunk x num_gradient_steps).
        print(f"{self._log_prefix} test-time inner loop: {inner_cfg.optimizer} "
              f"lr={inner_cfg.inner_lr_init} clip={inner_cfg.max_inner_grad_norm} "
              f"steps={infer_cfg.ttt_steps_per_chunk}x{inner_cfg.num_gradient_steps} "
              f"mc={inner_cfg.num_mc_samples}")
        # The anchoring route must match the one phi_0 was meta-trained under; the two put the
        # anchor latents in different places, so a mismatch conditions on tokens the
        # checkpoint never saw.
        if getattr(infer_cfg, "attention_anchor", False):
            print(f"{self._log_prefix} anchoring: attention prefix "
                  f"(k={infer_cfg.num_anchor_latent_frames} latents"
                  f"{' + sink' if infer_cfg.condition_on_first_frame_sink else ''}, "
                  f"no pinned frames, no boundary trim)")
        else:
            print(f"{self._log_prefix} anchoring: fused first-frame "
                  f"(condition_on_last_frame={infer_cfg.condition_on_last_frame}, "
                  f"k={infer_cfg.num_anchor_latent_frames}, "
                  f"sink={infer_cfg.condition_on_first_frame_sink})")

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        from ..pipelines.wan_video import WanVideoUnit_PromptEmbedder

        self.pipe.load_models_to_device(["text_encoder"])
        emb = WanVideoUnit_PromptEmbedder().encode_prompt(self.pipe, prompt)
        return emb.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)

    def _chunk_prompts(self, prompt, num_chunks: int) -> List[str]:
        """Per-chunk prompt schedule: entry k drives both the generation of chunk k and the
        memorize step that follows it. The base behaviour broadcasts one prompt to every
        chunk. Subclasses that vary the prompt across a narrative (see the memory-test
        driver) override *this* rather than reimplementing ``generate`` -- the chunk loop
        below carries the anchor/sink/k>1 bookkeeping, and a fork of it silently rots the
        moment that bookkeeping changes."""
        return [prompt] * num_chunks

    def generate(
        self,
        prompt,
        negative_prompt: str = "",
        *,
        input_image=None,
        seed: Optional[int] = None,
        extra_call_kwargs: Optional[dict] = None,
    ):
        """Generate one long video (list of PIL frames) for a single narrative."""
        icfg = self.infer_cfg
        prompts = self._chunk_prompts(prompt, icfg.num_chunks)
        # Only worth logging per chunk when the schedule actually varies (memory-test).
        varying_prompts = len(set(prompts)) > 1
        seed = icfg.seed if seed is None else seed
        extra_call_kwargs = extra_call_kwargs or {}

        # Reset the memory scratchpad to phi_0 for this narrative.
        restore_lora_state(self.pipe.dit, self.phi0)
        # Fresh inner-loop optimizer for this narrative, seeded from the just-restored
        # phi_0. Owned per video (not per chunk) so AdamW moments / Muon momentum carry
        # across the chunk sequence exactly as they do in meta-training, where
        # make_inner_optimizer is called once per outer step; and reset per video, matching
        # the per-video phi_0 reset in run_meta_inner_loop.
        updater = TestTimeInnerOptimizer(self.pipe.dit, self.inner_cfg)

        all_frames = []
        # Width of the local anchor block, in latent frames (k), and the number of decoded
        # frames of the preceding chunk it consumes. k=1 keeps the legacy single-frame anchor.
        k_anchor = max(1, int(getattr(icfg, "num_anchor_latent_frames", 1)))
        # Attention anchor (plain T2V DiTs): the previous chunk's trailing k latents are
        # prepended to the token sequence instead of overwriting this chunk's leading latents.
        # Everything the fused route needs -- pixel-frame handoff window, contiguous VAE
        # re-encode, boundary trimming -- falls away: the latents are handed forward directly
        # from the sampler, and every generated frame is new content.
        attn_anchor = bool(getattr(icfg, "attention_anchor", False))
        prev_latents = None
        sink_latent = None
        sink_wanted = bool(icfg.condition_on_first_frame_sink) and (
            attn_anchor or bool(icfg.condition_on_last_frame)
        )
        anchor_window = anchor_overlap_pixel_frames(k_anchor, sink_wanted)
        # Running image anchor for autoregressive frame conditioning. Seeded with the
        # optional I2V image; refreshed after each chunk to that chunk's trailing
        # `anchor_window` frames, which the pipeline re-encodes as ONE contiguous clip so the
        # anchor latents land at the positions they were trained at (see
        # WanVideoUnit_ImageEmbedderFused). For k=1 this is the single last frame, exactly as
        # before.
        cond_image = input_image
        cond_frames = None
        # Fixed global anchor ("first-frame sink"): the video's actual first raw frame,
        # captured once after chunk 0 and reused unchanged for every later chunk.
        sink_image = None
        for k in range(icfg.num_chunks):
            call_kwargs = dict(
                prompt=prompts[k],
                negative_prompt=negative_prompt,
                height=icfg.height,
                width=icfg.width,
                num_frames=icfg.frames_per_chunk,
                num_inference_steps=icfg.num_inference_steps,
                cfg_scale=icfg.cfg_scale,
                sigma_shift=icfg.sigma_shift,
                tiled=icfg.tiled,
                seed=seed + k,
            )
            # Attention anchor: prefix this chunk's tokens with the previous chunk's trailing
            # k latents (plus the sink). Straight from the sampler -- no decode, no re-encode,
            # so the anchor is byte-identical to what the previous chunk actually produced.
            if attn_anchor and prev_latents is not None:
                call_kwargs["anchor_latents"] = attention_anchor_latents(
                    prev_latents, k_anchor, sink_latent
                )
            # Image-condition this chunk (TI2V-5B fused first-frame latent):
            #   - k == 0: the optional I2V seed image, if any;
            #   - k  > 0: the last frame of the previous chunk, when enabled.
            if not attn_anchor and cond_image is not None and (k == 0 or icfg.condition_on_last_frame):
                call_kwargs["input_image"] = cond_image
            # k>1: hand the whole trailing window forward. `anchor_frames` supersedes
            # `input_image` inside the fused embedder, which encodes it as one clip. Only for
            # follow-up chunks -- chunk 0 has no preceding chunk, so it stays a plain
            # (optionally I2V-seeded) generation.
            if not attn_anchor and k > 0 and icfg.condition_on_last_frame and k_anchor > 1 and cond_frames is not None:
                call_kwargs["anchor_frames"] = cond_frames
            # Sink-condition follow-up chunks on the video's very first frame, alongside
            # the sliding local anchor above.
            sink_active = (
                not attn_anchor and k > 0 and icfg.condition_on_first_frame_sink
                and icfg.condition_on_last_frame and sink_image is not None
            )
            if sink_active:
                call_kwargs["sink_image"] = sink_image
            call_kwargs.update(extra_call_kwargs)

            # Latent handoff: ask the pipeline for the sampler's final latents alongside
            # the decoded frames. We memorize these latents directly (below) instead of
            # decode->re-encode, which skips a VAE encode per chunk and avoids the VAE
            # round-trip reconstruction error -- the memorized x0 is exactly what the
            # sampler produced.
            frames, chunk_latents = self.pipe(**call_kwargs, return_latents=True)
            # The leading frames of a follow-up chunk reproduce the pinned anchor(s) rather
            # than new content; drop them to avoid a duplicate-frame seam at the boundary.
            # Nothing to trim under the attention anchor: the prefix occupies its own
            # sequence positions, so every decoded frame of this chunk is new content.
            if not attn_anchor and k > 0 and icfg.condition_on_last_frame and icfg.drop_boundary_frame:
                emitted = frames[num_pinned_pixel_frames(num_clean_latents(k_anchor, sink_active)):]
            else:
                emitted = frames
            all_frames.extend(emitted)
            # Attention anchor: hand the sampler's latents forward directly, and capture the
            # video's first latent frame once as the fixed sink (a position-0 latent used at
            # position 0, exactly as meta-training slices it from chunk 0).
            if attn_anchor:
                prev_latents = chunk_latents.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
                if prev_latents.shape[2] < k_anchor:
                    raise ValueError(
                        f"chunk {k} produced {prev_latents.shape[2]} latent frames but the "
                        f"k={k_anchor} attention anchor needs {k_anchor}; raise "
                        f"frames_per_chunk or lower num_anchor_latent_frames."
                    )
                if k == 0 and sink_wanted:
                    sink_latent = prev_latents[:, :, 0:1].clone()
            # Carry the anchor forward. `anchor_window` frames for the k>1 block path (a
            # contiguous tail, re-encoded as one clip next iteration), the single last frame
            # for the legacy k=1 path.
            if not attn_anchor and icfg.condition_on_last_frame and len(frames) > 0:
                cond_image = frames[-1]
                cond_frames = frames[-anchor_window:] if k_anchor > 1 else None
                if cond_frames is not None and len(cond_frames) < anchor_window:
                    # Would silently encode to fewer than k anchor latents and condition the
                    # next chunk differently from training. Only reachable with a chunk
                    # shorter than the anchor window, i.e. a misconfigured frames_per_chunk.
                    raise ValueError(
                        f"chunk {k} produced {len(frames)} frames but the k={k_anchor} anchor "
                        f"block needs {anchor_window}; raise frames_per_chunk or lower "
                        f"num_anchor_latent_frames."
                    )
            # Capture the video's first raw frame once, as the fixed sink anchor.
            if k == 0 and sink_image is None and len(frames) > 0:
                sink_image = frames[0]
            print(f"{self._log_prefix} generated chunk {k + 1}/{icfg.num_chunks} "
                  f"({len(emitted)} frames)"
                  + (f" | prompt: {prompts[k][:40]}..." if varying_prompts else ""))

            if k == icfg.num_chunks - 1:
                continue

            # Memorize the chunk we just generated -- straight from the sampler's final
            # latents (no VAE decode->re-encode round-trip).
            x0 = chunk_latents.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            # Chunk k's own prompt -- the one it was generated under -- so the memorize
            # objective conditions on the same text as the sample it is fitting. Identical
            # to `prompt` unless a subclass supplies a varying schedule.
            context = self._encode_prompt(prompts[k])
            self.pipe.load_models_to_device(["dit"])
            for step in range(max(1, int(icfg.ttt_steps_per_chunk))):
                loss = ttt_update_inplace(
                    self.pipe, self.scheduler, x0, context, self.inner_cfg,
                    use_gradient_checkpointing=self.use_gradient_checkpointing,
                    updater=updater,
                )
                print(f"{self._log_prefix}  memorize chunk {k + 1} step {step + 1}/"
                      f"{icfg.ttt_steps_per_chunk} loss={loss:.6f}")

        # Leave the scratchpad at phi_0 for the next narrative.
        restore_lora_state(self.pipe.dit, self.phi0)
        return all_frames
