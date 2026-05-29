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

import torch, os, json, argparse, accelerate
from diffsynth.core import UnifiedDataset
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig, WanVideoUnit_PromptEmbedder
from diffsynth.diffusion import *
from diffsynth.diffusion.e2e_ttt import (
    InnerLoopConfig, ChunkingConfig, make_training_scheduler, run_meta_inner_loop,
    count_lora_params, enable_double_backward_attention,
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
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Second-order meta-training needs double-backward-capable attention.
        enable_double_backward_attention()
        self.chunk_cfg = ChunkingConfig(
            num_chunks=int(e2e_num_chunks),
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
        )
        self.truncate_steps = [int(s) for s in str(e2e_truncate_steps).split(",") if s != ""]
        # Dedicated 1000-step training scheduler (does not touch any inference scheduler).
        self.train_scheduler = make_training_scheduler(float(e2e_sigma_shift))
        print(f"[E2E-TTT] meta-training | LoRA params: {count_lora_params(self.pipe.dit):,} | "
              f"chunks={self.chunk_cfg.num_chunks} x {self.chunk_cfg.frames_per_chunk}f | "
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
        n = min(self.chunk_cfg.num_chunks, len(frames) // fpc)
        if n < 2:
            raise ValueError(
                f"Video has {len(frames)} frames but needs >= 2 chunks of {fpc} frames. "
                f"Increase --num_frames (>= {2 * fpc}) or lower --e2e_frames_per_chunk."
            )

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
            sub = frames[k * fpc:(k + 1) * fpc]
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

    def forward(self, data, inputs=None):
        chunk_latents, context = self._encode_chunks(data)
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
        )
        return meta_loss


def e2e_ttt_parser():
    parser = wan_parser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config supplying argument defaults (CLI flags still override).")
    g = parser.add_argument_group("E2E-TTT")
    g.add_argument("--e2e_num_chunks", type=int, default=3, help="Number of temporal chunks per video.")
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
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
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
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_tensorboard_log=args.enable_tensorboard_log,
        enable_swanlab_log=args.enable_swanlab_log,
        enable_wandb_log=args.enable_wandb_log,
    )
    launch_training_task(accelerator, dataset, model, model_logger, args=args)
