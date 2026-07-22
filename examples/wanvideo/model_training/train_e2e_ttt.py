"""
End-to-End Test-Time Training (E2E-TTT) meta-training entry point for Wan video models.

This reuses DiffSynth's standard training harness (UnifiedDataset, accelerate
``launch_training_task``, ModelLogger, PEFT LoRA injection). The only difference from
the vanilla ``train.py`` is the loss: instead of a single flow-matching SFT step,
``forward`` splits each video into a temporally-ordered chunk sequence and runs a
MAML-style memorize->predict inner loop (see ``diffsynth/diffusion/e2e_ttt.py``).

Because the inner loop never writes the adapted LoRA back into the real leaves
(``write_back=False``), the LoRA leaves stay at the meta-init phi_0, the meta-loss
back-propagates to phi_0, and the outer AdamW updates phi_0 directly -- so the
unmodified ``launch_training_task`` loop is correct with no W0-restore hook needed.

Targets: Wan2.1-T2V-1.3B and Wan2.2-TI2V-5B (single-DiT Wan pipelines).
"""

import torch, os, json, argparse, accelerate, datetime, resource
try:
    import psutil
except ImportError:
    psutil = None
from diffsynth.core import UnifiedDataset
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig, WanVideoUnit_PromptEmbedder
from diffsynth.diffusion import *
from diffsynth.diffusion.e2e_ttt import (
    InnerLoopConfig, ChunkingConfig, make_training_scheduler, run_meta_inner_loop,
    count_lora_params, enable_double_backward_attention, get_lora_params,
    ErrorRecycler,
)

# Reuse the vanilla module so all the model-loading / LoRA-injection plumbing is shared.
from train import WanTrainingModule, wan_parser

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WanE2ETTTTrainingModule(WanTrainingModule):
    def __init__(
        self,
        *args,
        e2e_num_chunks=3,
        e2e_frames_per_chunk=21,
        e2e_num_gradient_steps=1,
        e2e_num_mc_samples=2,
        e2e_inner_lr=5e-5,
        e2e_max_inner_grad_norm=1.0,
        e2e_inner_optimizer="sgd",
        e2e_truncate_steps="0",
        e2e_min_timestep_boundary=0.0,
        e2e_max_timestep_boundary=1.0,
        e2e_sigma_shift=5.0,
        e2e_algorithm="maml",
        e2e_first_order=False,
        e2e_condition_on_last_frame=True,
        e2e_use_error_recycling=False,
        e2e_num_grids=50,
        e2e_error_buffer_k=32,
        e2e_buffer_warmup_iter=20,
        e2e_noise_prob=0.9,
        e2e_latent_prob=0.9,
        e2e_y_prob=0.9,
        e2e_clean_prob=0.1,
        e2e_error_modulate_factor=0.0,
        e2e_anchor_sample_from_all_grids=True,
        outer_lr=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Resolve the meta-learning algorithm. --e2e_first_order is a back-compat alias
        # for FOMAML; --e2e_algorithm is the single source of truth otherwise.
        algorithm = str(e2e_algorithm).lower()
        if e2e_first_order and algorithm == "maml":
            algorithm = "fomaml"
        if algorithm not in ("maml", "fomaml", "reptile"):
            raise ValueError(f"--e2e_algorithm must be maml|fomaml|reptile, got {algorithm!r}")
        # Only exact second-order MAML needs double-backward-capable attention. FOMAML and
        # Reptile each do a single backward per inner step, so fused flash/sage kernels are fine.
        if algorithm == "maml":
            enable_double_backward_attention()
        # --- training-dynamics logging state ---
        # Previous-step LoRA snapshot, used to measure the *realized* per-step update
        # (||phi_t - phi_{t-1}|| / ||phi_t||) without needing post-backward grad access.
        self._prev_lora = None
        # Opt this module in to the runner's post-backward global grad-norm logging
        # (computed once grads are synced, written into self.log_metrics). The grad is the
        # OUTER (meta) gradient on phi_0, so name it with the "meta_" convention.
        self.log_grad_norm = True
        self.grad_norm_log_key = "train/meta_grad_norm"
        # Nominal outer-loop (meta) learning rate, shown in the startup banner. The value
        # actually logged each step (train/meta_lr) is the LIVE optimizer LR supplied by the
        # runner — see lr_log_key below — so it reflects any scheduler warmup, not this const.
        self.outer_lr = float(outer_lr) if outer_lr is not None else None
        # Keep every chart under the wandb "train" group: log the live outer LR as
        # "train/meta_lr" and drop the bare "loss" (the module's own "train/meta_loss" is
        # the same quantity). ModelLogger reads these two attributes; None suppresses.
        self.lr_log_key = "train/meta_lr"
        self.loss_log_key = None
        # num_chunks is an optional max cap: <=0 (or unset) -> adaptive, so the chunk
        # count is driven purely by each clip's length (len(frames) // frames_per_chunk).
        _num_chunks = int(e2e_num_chunks)
        self.chunk_cfg = ChunkingConfig(
            num_chunks=_num_chunks if _num_chunks > 0 else None,
            frames_per_chunk=int(e2e_frames_per_chunk),
        )
        self.inner_cfg = InnerLoopConfig(
            num_gradient_steps=int(e2e_num_gradient_steps),
            num_mc_samples=int(e2e_num_mc_samples),
            inner_lr_init=float(e2e_inner_lr),
            max_inner_grad_norm=float(e2e_max_inner_grad_norm),
            min_timestep_boundary=float(e2e_min_timestep_boundary),
            max_timestep_boundary=float(e2e_max_timestep_boundary),
            optimizer=str(e2e_inner_optimizer),
            algorithm=algorithm,
        )
        self.truncate_steps = [int(s) for s in str(e2e_truncate_steps).split(",") if s != ""]
        # Dedicated 1000-step training scheduler (does not touch any inference scheduler).
        self.train_scheduler = make_training_scheduler(float(e2e_sigma_shift))
        # First-frame (last-frame-of-previous-chunk) conditioning to match inference. This is
        # TI2V-5B's fused VAE first-frame mechanism, so it is only effective on a DiT that
        # supports it (fuse_vae_embedding_in_latents); requesting it on e.g. Wan2.1-T2V is a
        # no-op. When effective, chunks are sliced with a 1-frame overlap so each predict
        # chunk's first frame is the previous chunk's last frame (the inference anchor).
        _supports_fused = bool(getattr(self.pipe.dit, "fuse_vae_embedding_in_latents", False))
        self.condition_on_last_frame = bool(e2e_condition_on_last_frame) and _supports_fused
        if bool(e2e_condition_on_last_frame) and not _supports_fused:
            print("[E2E-TTT] NOTE: this DiT has no fused first-frame conditioning "
                  "(fuse_vae_embedding_in_latents); training without last-frame conditioning "
                  "(contiguous, non-overlapping chunks). This is expected for Wan2.1-T2V.")
        # SVI-style recycled-error buffers for anti-drift (arXiv:2510.09212). Per-process
        # only (no all_gather); the warmup iterations are collection-only, then recycled
        # errors are injected into the memorize inputs and the predict anchor frame.
        self.error_recycler = None
        if e2e_use_error_recycling:
            self.error_recycler = ErrorRecycler(
                num_grids=int(e2e_num_grids),
                error_buffer_k=int(e2e_error_buffer_k),
                buffer_warmup_iter=int(e2e_buffer_warmup_iter),
                noise_prob=float(e2e_noise_prob),
                latent_prob=float(e2e_latent_prob),
                y_prob=float(e2e_y_prob),
                clean_prob=float(e2e_clean_prob),
                error_modulate_factor=float(e2e_error_modulate_factor),
                sigma_shift=float(e2e_sigma_shift),
                anchor_sample_from_all_grids=bool(e2e_anchor_sample_from_all_grids),
            )
            print(f"[E2E-TTT] error recycling ON | grids={e2e_num_grids} x k={e2e_error_buffer_k} | "
                  f"warmup={e2e_buffer_warmup_iter} iters | probs noise/latent/y/clean="
                  f"{e2e_noise_prob}/{e2e_latent_prob}/{e2e_y_prob}/{e2e_clean_prob} | "
                  f"modulate={e2e_error_modulate_factor}")
        _chunks_desc = ("adaptive" if self.chunk_cfg.num_chunks is None
                        else f"<={self.chunk_cfg.num_chunks}")
        print(f"[E2E-TTT] meta-training | LoRA params: {count_lora_params(self.pipe.dit):,} | "
              f"chunks={_chunks_desc} x {self.chunk_cfg.frames_per_chunk}f | "
              f"condition_on_last_frame={self.condition_on_last_frame} | "
              f"outer_lr={self.outer_lr} | "
              f"algorithm={self.inner_cfg.algorithm} | "
              f"inner={self.inner_cfg.optimizer} lr={self.inner_cfg.inner_lr_init} "
              f"steps={self.inner_cfg.num_gradient_steps} mc={self.inner_cfg.num_mc_samples} | "
              f"truncate_steps={self.truncate_steps}")

    @torch.no_grad()
    def _encode_chunks(self, data):
        """Split one video into ordered sub-clips, VAE-encode each independently, and
        encode the (single) narrative prompt once. All chunks share the prompt."""
        pipe = self.pipe
        device, dtype = pipe.device, pipe.torch_dtype
        frames = data["video"]
        prompt = data["prompt"]

        fpc = self.chunk_cfg.frames_per_chunk
        # Chunk stride. With first-frame conditioning, consecutive chunks OVERLAP by one
        # frame (stride = fpc-1) so each chunk's first frame is the previous chunk's last
        # frame -- the same autoregressive anchor the inference generator pins to
        # (drop_boundary_frame). Without conditioning, chunks are contiguous (stride = fpc).
        stride = (fpc - 1) if self.condition_on_last_frame else fpc
        # Adaptive chunk count: as many chunks as fit in THIS clip. num_chunks (if set)
        # only caps it as a memory ceiling; leftover frames < fpc are dropped.
        n = (len(frames) - fpc) // stride + 1 if len(frames) >= fpc else 0
        if self.chunk_cfg.num_chunks is not None:
            n = min(self.chunk_cfg.num_chunks, n)
        if n < 2:
            # Too short to form even one memorize->predict pair. Signal the caller to
            # skip this video (rather than crash) so a mixed-length dataset trains fine.
            _need = fpc + stride  # frames for 2 chunks at this stride
            print(f"[E2E-TTT] skipping clip with {len(frames)} frames: needs >= 2 chunks "
                  f"of {fpc} frames at stride {stride} (>= {_need}).")
            return [], None

        # NOTE: pipe.load_models_to_device() is a no-op in the training path —
        # VRAM management is only wired up for inference (base_pipeline guards it
        # on self.vram_management_enabled, never set here). Without offload the
        # runner puts every frozen model on the GPU and leaves it there, so the
        # ~11.4 GB umt5-xxl text encoder would stay pinned through the entire
        # second-order DiT backward. The encoded latents/context are detached
        # before the DiT ever runs, so the encoder and VAE are dead weight during
        # run_meta_inner_loop — move them on/off explicitly so only the DiT holds
        # VRAM during the create_graph inner-loop grad.
        pipe.vae.to(device)
        chunk_latents = []
        for k in range(n):
            sub = frames[k * stride:k * stride + fpc]
            video = pipe.preprocess_video(sub)  # [1, C, T, H, W]
            z = pipe.vae.encode([video[0]], device=device, tiled=False)
            chunk_latents.append(z.to(dtype=dtype, device=device).detach())
        pipe.vae.to("cpu")

        pipe.text_encoder.to(device)
        context = WanVideoUnit_PromptEmbedder().encode_prompt(pipe, prompt)
        context = context.to(dtype=dtype, device=device).detach()
        pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        return chunk_latents, context

    @torch.no_grad()
    def _lora_dynamics_metrics(self):
        """Read-only LoRA training-dynamics metrics, computed on the meta-init phi_0
        *entering* this step. Returns a flat dict for ModelLogger. Never touches the
        meta-graph (no_grad + detach throughout)."""
        dit = self.pipe.dit
        out = {}

        # --- Effective injected-update ratio: ||dW||_F / ||W_base||_F per LoRA-wrapped
        # linear, where dW = scaling * (B @ A). B is zero-init, so this is exactly how
        # much the adapter has moved the frozen base weight. Aggregate mean/max + a
        # per-module-type breakdown (q/k/v/o/ffn.0/ffn.2).
        ratios, per_type = [], {}
        for name, mod in dit.named_modules():
            lora_A = getattr(mod, "lora_A", None)
            if lora_A is None or len(lora_A) == 0:
                continue
            adapter = next(iter(lora_A.keys()))            # PEFT adapter name (usually 'default')
            A = mod.lora_A[adapter].weight
            B = mod.lora_B[adapter].weight
            scale = mod.scaling[adapter]
            dW = scale * (B.float() @ A.float())
            base = mod.base_layer.weight.float()
            r = (dW.norm() / (base.norm() + 1e-8)).item()
            ratios.append(r)
            parts = name.split(".")
            key = parts[-1] if not parts[-1].isdigit() else f"{parts[-2]}.{parts[-1]}"
            per_type.setdefault(key, []).append(r)
        # phi_0 is updated by the OUTER optimizer, so these cumulative-change metrics use
        # the "meta_" naming convention.
        if ratios:
            out["lora/meta_dW_ratio_mean"] = sum(ratios) / len(ratios)
            out["lora/meta_dW_ratio_max"] = max(ratios)
            for k, v in per_type.items():
                out[f"lora/meta_dW_ratio/{k}"] = sum(v) / len(v)

        # --- Realized update-to-weight ratio: ||phi_t - phi_{t-1}|| / ||phi_t|| over all
        # LoRA params. This is the true optimizer step size (incl. lr/wd/Adam precond),
        # one step lagged. Skipped on the first step (no previous snapshot yet).
        cur = get_lora_params(dit)
        if self._prev_lora is not None:
            delta_sq, norm_sq = 0.0, 0.0
            for n, p in cur.items():
                pf = p.detach().float()
                norm_sq += pf.pow(2).sum().item()
                if n in self._prev_lora:
                    delta_sq += (pf - self._prev_lora[n]).pow(2).sum().item()
            out["lora/meta_update_ratio"] = (delta_sq ** 0.5) / ((norm_sq ** 0.5) + 1e-8)
        self._prev_lora = {n: p.detach().float().clone() for n, p in cur.items()}

        # --- GPU memory high-water mark (GB). Given the second-order backward parks at
        # the memory ceiling, this is the practical "how close to OOM" signal.
        if torch.cuda.is_available():
            out["gpu/mem_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
            out["gpu/mem_alloc_gb"] = torch.cuda.memory_allocated() / 1e9

        # --- CPU (process RSS) memory. Matters when offload streams frozen weights to host
        # RAM. mem_peak is the monotonic high-water mark (ru_maxrss, KB on Linux); mem_alloc
        # is the current resident set size.
        out["cpu/mem_peak_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        if psutil is not None:
            out["cpu/mem_alloc_gb"] = psutil.Process().memory_info().rss / 1e9
        return out

    def _zero_loss_skip(self):
        """Return a graph-connected zero scalar so a skipped (too-short) video is a
        no-op step: backward() deposits zero LoRA grads and the outer optimizer step
        applies no meaningful update. (With AdamW weight-decay a zero-grad step still
        applies negligible decoupled decay; skips are rare, so this is harmless.)"""
        zero = None
        for p in self.trainable_modules():
            term = p.float().sum() * 0.0
            zero = term if zero is None else zero + term
        if zero is None:  # no trainable params (shouldn't happen) -> detached zero
            zero = torch.zeros((), device=self.pipe.device, requires_grad=True)
        self.log_metrics = {"train/skipped_short_video": 1.0}
        return zero

    def forward(self, data, inputs=None):
        chunk_latents, context = self._encode_chunks(data)
        if len(chunk_latents) < 2:
            return self._zero_loss_skip()
        self.pipe.load_models_to_device(["dit"])
        meta_loss, stats = run_meta_inner_loop(
            self.pipe,
            self.train_scheduler,
            video_chunks=[chunk_latents],
            video_contexts=[context],
            inner_cfg=self.inner_cfg,
            truncate_steps=self.truncate_steps,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            write_back=False,
            algorithm=self.inner_cfg.algorithm,
            condition_on_first_frame=self.condition_on_last_frame,
            error_recycler=self.error_recycler,
        )
        # Surface inner-loop diagnostics so ModelLogger picks them up on the main process.
        # `meta_loss` is logged separately as the generic "loss"; these add the MAML-specific
        # breakdown (all detached — they must never touch the second-order meta-graph).
        # train/meta_lr (the live outer LR) is added by ModelLogger via lr_log_key, not here.
        # `meta_loss` is the scalar the runner back-propagates; for Reptile it is the
        # pseudo-gradient surrogate (an arbitrary inner-product value, not a meaningful
        # loss), so log train/meta_loss as 0 -- Reptile has no meta-objective, and the
        # memorize fit is already surfaced separately as train/memorize_loss. MAML/FOMAML
        # log their real predict loss. All detached -- never touch the meta-graph.
        is_reptile = str(getattr(self.inner_cfg, "algorithm", "maml")).lower() == "reptile"
        self.log_metrics = {
            "train/meta_loss": 0.0 if is_reptile else stats.get("monitor_loss", meta_loss.detach()),
            "train/memorize_loss": stats["memorize_loss"].detach() if torch.is_tensor(stats["memorize_loss"]) else stats["memorize_loss"],
            "train/num_pred_pairs": float(stats["num_pred"]),
            "train/num_mem_steps": float(stats["num_mem_steps"]),
            "train/inner_lr": float(self.inner_cfg.inner_lr_init),
            # Actual chunks used for THIS clip (adaptive: clip_frames // frames_per_chunk,
            # capped by num_chunks if set). Watch this to confirm the count varies per video.
            "train/num_chunks": float(len(chunk_latents)),
            "train/skipped_short_video": 0.0,
            # 1.0 when the predict loss conditions on the previous chunk's last frame
            # (TI2V-5B fused first-frame anchor), matching inference; 0.0 otherwise.
            "train/condition_on_last_frame": float(self.condition_on_last_frame),
        }
        # Recycled-error buffer fill/injection counters (per-process; rank 0 is logged).
        # Logged under their own top-level "err_buffer/" group (NOT the "train/" group).
        if self.error_recycler is not None:
            self.log_metrics.update(
                {k: v for k, v in self.error_recycler.stats().items()}
            )
        # Inner adaptation magnitude: how far the inner loop moved phi_0 per video.
        if "inner_adapt_norm" in stats:
            self.log_metrics["lora/inner_adapt_norm"] = stats["inner_adapt_norm"].detach()
            self.log_metrics["lora/inner_adapt_ratio"] = stats["inner_adapt_ratio"].detach()
        # ΔW ratio (mean/max/per-type), realized update-to-weight ratio, GPU mem.
        self.log_metrics.update(self._lora_dynamics_metrics())
        return meta_loss


def e2e_ttt_parser():
    parser = wan_parser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config supplying argument defaults (CLI flags still override).")
    parser.add_argument("--frame_rate", type=int, default=24,
                        help="Target frame rate (fps) for sampling video frames in the data operator.")
    g = parser.add_argument_group("E2E-TTT")
    g.add_argument("--e2e_num_chunks", type=int, default=0,
                   help="Optional max temporal chunks per video (memory ceiling). "
                        "<=0 means adaptive: one chunk per --e2e_frames_per_chunk frames in the clip.")
    g.add_argument("--e2e_frames_per_chunk", type=int, default=21, help="Frames per chunk (4n+1).")
    g.add_argument("--e2e_num_gradient_steps", type=int, default=1, help="Inner-loop SGD steps per memorize chunk.")
    g.add_argument("--e2e_num_mc_samples", type=int, default=2, help="Monte-Carlo (t, noise) samples per flow-loss eval.")
    g.add_argument("--e2e_inner_lr", type=float, default=5e-5, help="Inner-loop learning rate.")
    g.add_argument("--e2e_max_inner_grad_norm", type=float, default=1.0, help="Per-tensor inner-grad clip.")
    g.add_argument("--e2e_inner_optimizer", type=str, default="sgd", help="Inner optimizer: sgd|adamw|muon|muonclip.")
    g.add_argument("--e2e_truncate_steps", type=str, default="0", help="Comma-separated inner steps to truncate BPTT at.")
    g.add_argument("--e2e_min_timestep_boundary", type=float, default=0.0, help="Min timestep fraction for inner loss.")
    g.add_argument("--e2e_max_timestep_boundary", type=float, default=1.0, help="Max timestep fraction for inner loss.")
    g.add_argument("--e2e_sigma_shift", type=float, default=5.0, help="Flow-matching sigma shift for the TTT scheduler.")
    g.add_argument("--e2e_algorithm", type=str, default="maml", choices=["maml", "fomaml", "reptile"],
                   help="Meta-learning algorithm: maml (second-order), fomaml (first-order MAML), "
                        "or reptile (SGD adaptation then move phi_0 toward the adapted weights). "
                        "fomaml/reptile use a single backward (fused attention OK) and ignore "
                        "--e2e_truncate_steps.")
    g.add_argument("--e2e_first_order", action="store_true",
                   help="Back-compat alias for --e2e_algorithm fomaml (drop the Hessian term).")
    g.add_argument("--e2e_condition_on_last_frame", type=lambda s: str(s).lower() not in ("0", "false", "no"),
                   default=True,
                   help="Condition each predict chunk on the previous chunk's last frame, "
                        "matching TI2V-5B inference (fused first-frame anchor + 1-frame chunk "
                        "overlap). Effective only on a DiT with fused first-frame conditioning "
                        "(TI2V-5B); a no-op otherwise. Set false to train without it.")
    er = parser.add_argument_group("E2E-TTT error recycling (SVI-style anti-drift)")
    er.add_argument("--e2e_use_error_recycling", type=lambda s: str(s).lower() not in ("0", "false", "no"),
                    default=False,
                    help="Enable SVI-style recycled-error buffers (arXiv:2510.09212): harvest the "
                         "model's own prediction errors into timestep-bucketed buffers and inject "
                         "them into the memorize inputs / predict anchor frame, with clean targets.")
    er.add_argument("--e2e_num_grids", type=int, default=50,
                    help="Number of timestep grids (buckets), matched to a simulated "
                         "num_grids-step inference schedule (SVI: --num_grids).")
    er.add_argument("--e2e_error_buffer_k", type=int, default=250,
                    help="Max error samples per grid per buffer (SVI: --error_buffer_k, theirs 500). "
                         "CPU RAM = 2 buffers x num_grids x k x chunk-latent size (~2 MB each for "
                         "TI2V-5B 21f 704x1280).")
    er.add_argument("--e2e_buffer_warmup_iter", type=int, default=20,
                    help="Collection-only outer steps before injection starts (SVI's "
                         "--buffer_warmup_iter, but WITHOUT the all_gather: buffers are strictly "
                         "per-process; E2E-TTT harvests num_mem_steps x num_mc_samples errors per "
                         "step, so local collection fills the grids quickly).")
    er.add_argument("--e2e_noise_prob", type=float, default=0.9,
                    help="Predict loss: probability of injecting a recycled noise-direction error "
                         "into the noise, with a CLEAN target (SVI: --noise_prob).")
    er.add_argument("--e2e_latent_prob", type=float, default=0.9,
                    help="Memorize loss: probability of corrupting the memorized chunk "
                         "CONSISTENTLY (input and target), replicating the test-time TTT update. "
                         "Also gates the predict-loss latent injection (clean target) "
                         "(SVI: --latent_prob).")
    er.add_argument("--e2e_y_prob", type=float, default=0.9,
                    help="Probability of injecting a recycled data-direction error into the "
                         "pinned anchor frame of the predict loss (SVI: --y_prob).")
    er.add_argument("--e2e_clean_prob", type=float, default=0.1,
                    help="Probability of overriding a draw to fully clean inputs "
                         "(SVI: --clean_prob).")
    er.add_argument("--e2e_error_modulate_factor", type=float, default=0.0,
                    help="Injected errors are scaled by uniform(1-f, 1+f) "
                         "(SVI: --error_modulate_factor).")
    er.add_argument("--e2e_anchor_sample_from_all_grids",
                    type=lambda s: str(s).lower() not in ("0", "false", "no"), default=True,
                    help="Sample the anchor error from ALL timestep grids rather than the "
                         "current one (SVI: --y_error_sample_from_all_grids). The anchor drift "
                         "is a terminal, data-space error not tied to the current denoising "
                         "timestep, so pooling grids maximises the small anchor buffer's use.")
    return parser


def apply_yaml_config(parser, config_path):
    """Load a YAML config and use its values as argument defaults so a single
    `--config foo.yaml` replaces the long CLI. The YAML may be grouped into arbitrary
    sections (e.g. `model:`, `dataset:`, `train:`, `lora:`, `e2e_ttt:`) for readability;
    only leaf keys matter, and they must match argparse dest names (e.g. `lora_rank`,
    `e2e_num_chunks`). CLI flags passed on the command line still override the YAML.
    Returns the set of unrecognised leaf keys (for a warning)."""
    import yaml

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    flat = {}
    def _walk(d):
        for k, v in d.items():
            if isinstance(v, dict):
                _walk(v)
            else:
                flat[k] = v
    _walk(cfg)

    # Coerce a few fields to the string/JSON forms the argparse layer expects.
    if isinstance(flat.get("model_paths"), list):
        flat["model_paths"] = json.dumps(flat["model_paths"])
    if isinstance(flat.get("lora_target_modules"), list):
        flat["lora_target_modules"] = ",".join(map(str, flat["lora_target_modules"]))
    if isinstance(flat.get("data_file_keys"), list):
        flat["data_file_keys"] = ",".join(map(str, flat["data_file_keys"]))
    if isinstance(flat.get("e2e_truncate_steps"), list):
        flat["e2e_truncate_steps"] = ",".join(map(str, flat["e2e_truncate_steps"]))

    valid_dests = {a.dest for a in parser._actions}
    defaults = {k: v for k, v in flat.items() if k in valid_dests}
    parser.set_defaults(**defaults)
    # A value supplied by the YAML satisfies a `required=True` argument, but argparse
    # enforces `required` regardless of defaults, so clear it for those args.
    for action in parser._actions:
        if action.dest in defaults and getattr(action, "required", False):
            action.required = False
    return set(flat) - valid_dests - {"config"}


if __name__ == "__main__":
    parser = e2e_ttt_parser()
    # Pre-parse only --config so the YAML can populate defaults before the real parse.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()
    if pre_args.config is not None:
        unknown = apply_yaml_config(parser, pre_args.config)
        if unknown:
            print(f"[E2E-TTT] WARNING: ignoring unknown config keys: {sorted(unknown)}")
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    # Give each run its own timestamped output dir so reruns don't overwrite the previous
    # run's epoch-*.safetensors / step-*.safetensors / wandb logs. The timestamp is generated
    # on the main process and broadcast so every rank in a multi-GPU run agrees on one folder
    # (otherwise each rank would stamp its own time and the saves would scatter). The suffix
    # keeps the basename self-describing, so the default wandb run name stays meaningful.
    run_stamp = [datetime.datetime.now().strftime("%Y%m%d-%H%M%S")] if accelerator.is_main_process else [None]
    accelerate.utils.broadcast_object_list(run_stamp, from_process=0)
    args.output_path = f"{os.path.normpath(args.output_path)}-{run_stamp[0]}"
    if accelerator.is_main_process:
        print(f"[E2E-TTT] Run outputs -> {args.output_path}")
    # Spatial alignment divisor = VAE spatial compression x DiT patch size (2).
    # The Wan2.1 VAE is 8x (-> divisor 16); the Wan2.2 VAE shipped with TI2V-5B is
    # 16x (-> divisor 32). A frame whose height is a multiple of 16 but not 32 (e.g.
    # 720) yields an odd latent height (45) under the 16x VAE, and the DiT patchify
    # (stride-2 conv) silently floors it to 44 -> target/noise_pred size mismatch.
    _paths_str = str(args.model_paths) + str(getattr(args, "tokenizer_path", ""))
    _is_wan22 = "Wan2.2" in _paths_str or "TI2V-5B" in _paths_str
    _spatial_div = 32 if _is_wan22 else 16
    # ImageCropAndResize only snaps to the division factor when height/width are left
    # unset; with explicit sizes (our case) it crops verbatim. So snap them here too,
    # otherwise an explicit 720 stays 720 and re-triggers the odd-latent mismatch.
    _aligned_h = args.height // _spatial_div * _spatial_div
    _aligned_w = args.width // _spatial_div * _spatial_div
    if accelerator.is_main_process:
        _vae_desc = "Wan2.2 VAE 16x" if _is_wan22 else "Wan2.1 VAE 8x"
        print(f"[E2E-TTT] spatial division factor = {_spatial_div} ({_vae_desc})")
        if (_aligned_h, _aligned_w) != (args.height, args.width):
            print(f"[E2E-TTT] snapped frame size {args.height}x{args.width} -> "
                  f"{_aligned_h}x{_aligned_w} to align with the DiT patch grid")
    args.height, args.width = _aligned_h, _aligned_w
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=_spatial_div,
            width_division_factor=_spatial_div,
            # num_frames is now an optional UPPER bound; <=0 (or unset) -> load each
            # clip's full length so the adaptive chunker sees the true video duration.
            num_frames=(args.num_frames if args.num_frames and args.num_frames > 0 else None),
            time_division_factor=4,
            time_division_remainder=1,
            frame_rate=args.frame_rate
        ),
    )
    model = WanE2ETTTTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        task=args.task,
        device="cpu" if (args.initialize_model_on_cpu or args.enable_model_cpu_offload) else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        e2e_num_chunks=args.e2e_num_chunks,
        e2e_frames_per_chunk=args.e2e_frames_per_chunk,
        e2e_num_gradient_steps=args.e2e_num_gradient_steps,
        e2e_num_mc_samples=args.e2e_num_mc_samples,
        e2e_inner_lr=args.e2e_inner_lr,
        e2e_max_inner_grad_norm=args.e2e_max_inner_grad_norm,
        e2e_inner_optimizer=args.e2e_inner_optimizer,
        e2e_truncate_steps=args.e2e_truncate_steps,
        e2e_min_timestep_boundary=args.e2e_min_timestep_boundary,
        e2e_max_timestep_boundary=args.e2e_max_timestep_boundary,
        e2e_sigma_shift=args.e2e_sigma_shift,
        e2e_algorithm=args.e2e_algorithm,
        e2e_first_order=args.e2e_first_order,
        e2e_condition_on_last_frame=args.e2e_condition_on_last_frame,
        e2e_use_error_recycling=args.e2e_use_error_recycling,
        e2e_num_grids=args.e2e_num_grids,
        e2e_error_buffer_k=args.e2e_error_buffer_k,
        e2e_buffer_warmup_iter=args.e2e_buffer_warmup_iter,
        e2e_noise_prob=args.e2e_noise_prob,
        e2e_latent_prob=args.e2e_latent_prob,
        e2e_y_prob=args.e2e_y_prob,
        e2e_clean_prob=args.e2e_clean_prob,
        e2e_error_modulate_factor=args.e2e_error_modulate_factor,
        e2e_anchor_sample_from_all_grids=args.e2e_anchor_sample_from_all_grids,
        outer_lr=args.learning_rate,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_tensorboard_log=args.enable_tensorboard_log,
        enable_swanlab_log=args.enable_swanlab_log,
        enable_wandb_log=args.enable_wandb_log,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_entity=args.wandb_entity,
        config=vars(args),
    )
    launch_training_task(accelerator, dataset, model, model_logger, args=args)
