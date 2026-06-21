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
    dit=None,
) -> torch.Tensor:
    """Rectified flow-matching loss matching DiffSynth's ``FlowMatchSFTLoss``.

    ``noise_pred = model_fn(z_t, t, c)`` and target ``= noise - x0``; averaged over
    ``num_mc_samples`` random (t, noise) draws.
    """
    if x0.dim() != 5:
        raise ValueError(f"Expected x0 [B,C,T,H,W], got {tuple(x0.shape)}")
    dit = dit if dit is not None else pipe.dit
    device, dtype = pipe.device, pipe.torch_dtype
    num_ts = len(scheduler.timesteps)
    min_ts = int(inner_cfg.min_timestep_boundary * num_ts)
    max_ts = max(min_ts + 1, int(inner_cfg.max_timestep_boundary * num_ts))

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
        latents = scheduler.add_noise(x0, noise, timestep)
        target = scheduler.training_target(x0, noise, timestep)
        noise_pred = _model_fn_with_override(
            pipe, dit, params_override,
            latents=latents, timestep=timestep, context=context,
            use_gradient_checkpointing=gc,
        )
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
    """
    algorithm = (algorithm or getattr(inner_cfg, "algorithm", "maml")).lower()
    if algorithm == "reptile":
        return _run_reptile_inner_loop(
            pipe, scheduler, video_chunks, video_contexts, inner_cfg,
            learned_lrs=learned_lrs,
            use_gradient_checkpointing=use_gradient_checkpointing,
            write_back=write_back,
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

        for k in range(len(chunks) - 1):
            # ---- memorize chunk k ----
            for _ in range(max(1, int(inner_cfg.num_gradient_steps))):
                params_override = {**current_lora}
                loss_mem = compute_flow_matching_loss(
                    pipe, scheduler, chunks[k], ctx, inner_cfg,
                    params_override=params_override,
                    use_gradient_checkpointing=use_gradient_checkpointing,
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
            loss_pred = compute_flow_matching_loss(
                pipe, scheduler, chunks[k + 1], ctx, inner_cfg,
                params_override={**current_lora},
                use_gradient_checkpointing=use_gradient_checkpointing,
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


def ttt_update_inplace(
    pipe,
    scheduler: FlowMatchScheduler,
    x0: torch.Tensor,
    context: torch.Tensor,
    inner_cfg: InnerLoopConfig,
    *,
    use_gradient_checkpointing: bool = False,
) -> float:
    """One or more first-order LoRA updates applied in place (no second-order graph).
    Used between clips at test time. Returns the last loss value."""
    lora_params = get_trainable_lora_params(pipe.dit)
    if not lora_params:
        raise ValueError("No trainable LoRA params found for test-time TTT.")
    names = list(lora_params.keys())
    params = [lora_params[n] for n in names]
    last = 0.0
    for _ in range(max(1, int(inner_cfg.num_gradient_steps))):
        with torch.enable_grad():
            loss = compute_flow_matching_loss(
                pipe, scheduler, x0, context, inner_cfg,
                params_override=None,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )
            grads = torch.autograd.grad(loss, params, allow_unused=True)
        with torch.no_grad():
            for p, g in zip(params, grads):
                if g is None:
                    continue
                if inner_cfg.max_inner_grad_norm > 0:
                    gn = g.norm()
                    if gn > inner_cfg.max_inner_grad_norm:
                        g = g * (inner_cfg.max_inner_grad_norm / (gn + 1e-8))
                p.sub_(inner_cfg.inner_lr_init * g)
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
    """

    def __init__(
        self,
        pipe,
        inner_cfg: InnerLoopConfig,
        infer_cfg: InferenceConfig,
        *,
        phi0: Optional[Dict[str, torch.Tensor]] = None,
        use_gradient_checkpointing: bool = False,
    ):
        self.pipe = pipe
        self.inner_cfg = inner_cfg
        self.infer_cfg = infer_cfg
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.scheduler = make_training_scheduler(infer_cfg.sigma_shift)
        self.phi0 = phi0 if phi0 is not None else snapshot_lora_state(pipe.dit)

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        from ..pipelines.wan_video import WanVideoUnit_PromptEmbedder

        self.pipe.load_models_to_device(["text_encoder"])
        emb = WanVideoUnit_PromptEmbedder().encode_prompt(self.pipe, prompt)
        return emb.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)

    @torch.no_grad()
    def _encode_video_to_latents(self, frames) -> torch.Tensor:
        self.pipe.load_models_to_device(["vae"])
        video = self.pipe.preprocess_video(frames)  # [1, C, T, H, W] in [-1, 1]
        latents = self.pipe.vae.encode(
            [video[0]], device=self.pipe.device,
            tiled=self.infer_cfg.tiled,
        ).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        return latents

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        *,
        input_image=None,
        seed: Optional[int] = None,
        extra_call_kwargs: Optional[dict] = None,
    ):
        """Generate one long video (list of PIL frames) for a single narrative."""
        icfg = self.infer_cfg
        seed = icfg.seed if seed is None else seed
        extra_call_kwargs = extra_call_kwargs or {}

        # Reset the memory scratchpad to phi_0 for this narrative.
        restore_lora_state(self.pipe.dit, self.phi0)

        all_frames = []
        # Running image anchor for autoregressive frame conditioning. Seeded with the
        # optional I2V image; refreshed to the last generated frame after each chunk.
        cond_image = input_image
        for k in range(icfg.num_chunks):
            call_kwargs = dict(
                prompt=prompt,
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
            # Image-condition this chunk (TI2V-5B fused first-frame latent):
            #   - k == 0: the optional I2V seed image, if any;
            #   - k  > 0: the last frame of the previous chunk, when enabled.
            if cond_image is not None and (k == 0 or icfg.condition_on_last_frame):
                call_kwargs["input_image"] = cond_image
            call_kwargs.update(extra_call_kwargs)

            frames = self.pipe(**call_kwargs)
            # The first frame of a follow-up chunk reproduces the anchor frame; drop it
            # to avoid a duplicate-frame seam at the chunk boundary.
            if k > 0 and icfg.condition_on_last_frame and icfg.drop_boundary_frame:
                emitted = frames[1:]
            else:
                emitted = frames
            all_frames.extend(emitted)
            # Carry the last frame forward as the next chunk's anchor.
            if icfg.condition_on_last_frame and len(frames) > 0:
                cond_image = frames[-1]
            print(f"[E2E-TTT] generated chunk {k + 1}/{icfg.num_chunks} "
                  f"({len(emitted)} frames)")

            if k == icfg.num_chunks - 1:
                continue

            # Memorize the chunk we just generated.
            x0 = self._encode_video_to_latents(frames)
            context = self._encode_prompt(prompt)
            self.pipe.load_models_to_device(["dit"])
            for step in range(max(1, int(icfg.ttt_steps_per_chunk))):
                loss = ttt_update_inplace(
                    self.pipe, self.scheduler, x0, context, self.inner_cfg,
                    use_gradient_checkpointing=self.use_gradient_checkpointing,
                )
                print(f"[E2E-TTT]  memorize chunk {k + 1} step {step + 1}/"
                      f"{icfg.ttt_steps_per_chunk} loss={loss:.6f}")

        # Leave the scratchpad at phi_0 for the next narrative.
        restore_lora_state(self.pipe.dit, self.phi0)
        return all_frames
