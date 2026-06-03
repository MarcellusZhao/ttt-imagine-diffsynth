# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository identity

This is a fork of **ModelScope DiffSynth-Studio** (`diffsynth` package, v2.x — see [pyproject.toml](pyproject.toml)). The directory is named `ttt-imagine-diffsynth` because it is vendored alongside the `ttt-imagine` (VideoTuna) project. The bulk of the code is upstream DiffSynth-Studio — treat it as a Diffusion model engine for inference and training of FLUX, Wan, Qwen-Image, Z-Image, LTX, HiDream, ACE-Step, Anima, MOVA, etc.

**Local addition — End-to-End Test-Time Training (E2E-TTT):** ported from the reference in the sibling `ttt-imagine` repo (`e2e_ttt_video/`) into DiffSynth-native form for `Wan2.1-T2V-1.3B` and `Wan2.2-TI2V-5B`. See the E2E-TTT section below.

Origin: `git@github.com:MarcellusZhao/ttt-imagine-diffsynth.git`. Upstream docs: https://diffsynth-studio-doc.readthedocs.io/en/latest/

## Install

```
pip install -e .
```

Python ≥ 3.10.1. There are no unit tests and no lint/CI beyond a PyPI publish workflow ([.github/workflows/publish.yaml](.github/workflows/publish.yaml)) — "tests" in practice are running the example scripts under [examples/](examples/) end-to-end against downloaded model weights. Optional extras: `[npu]`, `[npu_aarch64]`, `[audio]`, `[all]`.

## Running things

Every model family has its own folder under `examples/<family>/` with the same four-part structure:
- `model_inference/<ModelName>.py` — standalone, single-GPU inference scripts. Run with `python examples/<family>/model_inference/<ModelName>.py`. (The IDE-open file [examples/wanvideo/model_inference/Wan2.1-T2V-1.3B.py](examples/wanvideo/model_inference/Wan2.1-T2V-1.3B.py) is the canonical minimal Wan T2V example.)
- `model_inference_low_vram/<ModelName>.py` — same model with VRAM management / offload enabled.
- `model_training/train.py` — the training entrypoint for that family (one file per family, dispatched via argparse).
- `model_training/{full,lora}/<ModelName>.sh` — shell launchers that call `accelerate launch examples/<family>/model_training/train.py …` with the right flags and `modelscope download …` for the example dataset.

Pick the existing `.sh` for the variant you want and adapt; do **not** invent new argparse flags without checking [diffsynth/diffusion/parsers.py](diffsynth/diffusion/parsers.py) — the parser composition (`add_general_config`, `add_video_size_config`, etc.) is the source of truth for available flags.

Models are downloaded from ModelScope by default. Outside China set `os.environ["MODELSCOPE_DOMAIN"] = "www.modelscope.ai"`, or switch source with `DIFFSYNTH_DOWNLOAD_SOURCE` (see [docs/en/Pipeline_Usage/Environment_Variables.md](docs/en/Pipeline_Usage/Environment_Variables.md)).

## Architecture: the three layers

The repo is organized around **one pattern repeated per model family**. Understanding the layers is the fastest way to navigate:

### 1. `diffsynth/models/` — pure `nn.Module`s

One file per architecture component (DiT, VAE, text encoder, image encoder, controlnet, adapter, …). File naming is `<family>_<component>.py` (e.g. `wan_video_dit.py`, `wan_video_vae.py`, `flux_text_encoder_t5.py`). No pipeline logic here.

Loading is registry-driven: [diffsynth/configs/model_configs.py](diffsynth/configs/model_configs.py) maps a `model_hash` (hash of weight file keys) → `model_name` → `model_class` (+ optional `state_dict_converter` under [diffsynth/utils/state_dict_converters/](diffsynth/utils/state_dict_converters/)). `ModelConfig(model_id=..., origin_file_pattern=...)` resolves to a snapshot download + hash lookup. To add a new model architecture, add a class under `models/`, register the hash in `configs/model_configs.py`, and (if weights need remapping) write a converter.

### 2. `diffsynth/pipelines/<family>.py` — `BasePipeline` + `PipelineUnit` sequences

Each pipeline subclasses `BasePipeline` ([diffsynth/diffusion/base_pipeline.py](diffsynth/diffusion/base_pipeline.py)) and exposes `from_pretrained(model_configs=[...], ...)` and `__call__(**inputs)`. The pipeline's behavior is **a list of `PipelineUnit`s** stored in `self.units` (see e.g. `WanVideoPipeline.units` in [diffsynth/pipelines/wan_video.py:55](diffsynth/pipelines/wan_video.py#L55)). Each unit declares `input_params` / `output_params` and a `process()` method; the runner threads a kwargs dict through them. This is also how the training module reuses inference logic — `WanTrainingModule.forward` in [examples/wanvideo/model_training/train.py](examples/wanvideo/model_training/train.py) iterates `self.pipe.units` and calls the loss only at the end.

When adding a new conditioning modality / control / adapter, the right move is almost always **a new `PipelineUnit` inserted into `self.units`**, not new branches inside `__call__`.

### 3. `diffsynth/diffusion/` — training harness

- [training_module.py](diffsynth/diffusion/training_module.py) — `DiffusionTrainingModule` base class. Handles model loading via `parse_model_configs`, LoRA injection (`add_lora_to_model`, `auto_detect_lora_target_modules`), fp8/offload model selection, gradient checkpointing toggles, checkpoint resume, and `split_pipeline_units` for split (data-process / train) tasks.
- [runner.py](diffsynth/diffusion/runner.py) — `launch_training_task` (the standard accelerate loop) and `launch_data_process_task` (precompute embeddings/latents to disk). Both consume the same `DiffusionTrainingModule`. Task strings (`sft`, `sft:data_process`, `sft:train`, `direct_distill`, …) are dispatched in each family's `train.py`.
- [loss.py](diffsynth/diffusion/loss.py) — `FlowMatchSFTLoss`, `DirectDistillLoss`, etc.
- [parsers.py](diffsynth/diffusion/parsers.py) — composable argparse fragments. `add_general_config(parser)` is the standard bundle used by every family's `train.py`.
- [flow_match.py](diffsynth/diffusion/flow_match.py) — `FlowMatchScheduler`.

Per-family training scripts ([examples/<family>/model_training/train.py](examples/wanvideo/model_training/train.py)) subclass `DiffusionTrainingModule`, override `get_pipeline_inputs(data)` to map dataset rows → pipeline kwargs, and wire `task → loss` in `self.task_to_loss`.

## VRAM / offload model

Two distinct mechanisms — don't confuse them:

- **Inference VRAM management** ([diffsynth/core/vram/](diffsynth/core/vram/), [diffsynth/configs/vram_management_module_maps.py](diffsynth/configs/vram_management_module_maps.py)) — per-`ModelConfig` `offload_device`/`onload_device`/`offload_dtype`/`onload_dtype` fields move weights between CPU↔GPU and dtype (fp8/bf16) **layer-by-layer**. Supports disk offload via `load_model_with_disk_offload`. Configured at `from_pretrained` time.
- **Training CPU offload** ([diffsynth/core/offload_training/](diffsynth/core/offload_training/), `OffloadTrainingManager`) — enabled by `--enable_model_cpu_offload` on training scripts. Streams weights layer-by-layer during forward/backward so consumer GPUs can LoRA-train 14B models. Single-GPU only. `--enable_optimizer_cpu_offload` and `--cpu_offload_split_threshold` are sub-knobs of this.

`fp8_models` and `offload_models` argparse args take comma-separated component names (`dit,text_encoder,…`) — they apply to frozen/non-trainable models only, gradient-carrying paths must stay in bf16.

## Data pipeline

[diffsynth/core/data/unified_dataset.py](diffsynth/core/data/unified_dataset.py) — `UnifiedDataset` reads `metadata.csv` (or json) under a base path, applies a chain of **operators** ([diffsynth/core/data/operators.py](diffsynth/core/data/operators.py): `LoadVideo`, `LoadAudio`, `ImageCropAndResize`, `ToAbsolutePath`, …) composed with the `>>` operator. `--data_file_keys` (default `"image,video"`) declares which columns are file paths. Custom columns get custom operators via `special_operator_map` (see Wan's `train.py` for the `animate_face_video` / `input_audio` / `wantodance_music_path` examples).

`UnifiedDataset.default_video_operator(...)` is the standard video pipeline (handles `--height`, `--width`, `--max_pixels`, `--num_frames`, division factors). Frame/time division factors are model-specific (Wan uses 16/16/4 with remainder 1; WanToDance global uses 1/1).

## Conventions worth knowing

- **`remove_prefix_in_ckpt`** — saved LoRA / full checkpoints strip a prefix like `pipe.dit.` so they can be reloaded as standalone weights. Match the prefix the upstream model expects.
- **`lora_base_model` + `lora_target_modules`** — LoRA is added to one named submodule of the pipe (e.g. `dit`) with module name patterns (e.g. `q,k,v,o,ffn.0,ffn.2`). `auto_detect_lora_target_modules` handles cases where the right targets aren't obvious.
- **Split training (`task=sft:data_process` → `task=sft:train`)** — for very large models, first run `data_process` to dump text/VAE embeddings to disk, then run `train` with the cached embeddings to avoid re-encoding each step.
- **Diffusion Templates** ([diffsynth/diffusion/template.py](diffsynth/diffusion/template.py)) — plugin framework loaded from a directory containing a `model.py`. Used by image-to-LoRA models and other custom controllable-generation plugins.
- **NPU support** — [diffsynth/core/device/npu_compatible_device.py](diffsynth/core/device/npu_compatible_device.py) + [diffsynth/core/npu_patch/](diffsynth/core/npu_patch/) wrap device selection. Use `get_device_type()` instead of hardcoding `"cuda"` when adding new pipelines.
- **Sequence parallel** — `enable_usp()` on pipelines swaps attention/forward methods via xfuser ([diffsynth/utils/xfuser/](diffsynth/utils/xfuser/)). See [examples/wanvideo/acceleration/unified_sequence_parallel.py](examples/wanvideo/acceleration/unified_sequence_parallel.py).

## End-to-End Test-Time Training (E2E-TTT)

A DiffSynth-native port of the `e2e_ttt_video/` algorithm from the sibling `ttt-imagine` repo. Long videos are generated chunk-by-chunk; the model adapts to its own preceding chunks via a LoRA "memory scratchpad", meta-trained MAML-style so a few inner-loop steps generalize to the *next* chunk.

- **Core module** — [diffsynth/diffusion/e2e_ttt.py](diffsynth/diffusion/e2e_ttt.py): config dataclasses (`InnerLoopConfig`, `ChunkingConfig`, `InferenceConfig`), differentiable inner-loop optimizers (`DifferentiableSGD/AdamW/Muon/MuonClip`, `MetaLearnedLRSchedule`), LoRA-state helpers, the rectified-flow loss (reuses DiffSynth's `FlowMatchScheduler` + `pipe.model_fn`), `run_meta_inner_loop` (the memorize→predict dispatcher for MAML/FOMAML/Reptile — see `--e2e_algorithm` below; Reptile routes to `_run_reptile_inner_loop`), the first-order test-time `ttt_update_inplace`, and `WanE2ETTTSequentialGenerator`.
- **Meta-training** — [examples/wanvideo/model_training/train_e2e_ttt.py](examples/wanvideo/model_training/train_e2e_ttt.py) subclasses `WanTrainingModule`; its `forward` splits each video into chunks and returns a single scalar the stock `launch_training_task` back-propagates (no W0-restore hook needed: `write_back=False` keeps LoRA leaves at φ₀ so the outer AdamW updates φ₀ directly). The `--e2e_algorithm` flag selects the meta-learning variant — all three return a scalar whose `.backward()` deposits the right φ₀ gradient, so the outer loop is identical across them:
  - `maml` (default) — exact second-order MAML; scalar = mean next-chunk **predict** loss (`create_graph=True`).
  - `fomaml` — first-order MAML; same predict-loss scalar, Hessian dropped (`create_graph=False`); grad reaches φ₀ only via the clone→φ₀ identity path. `--e2e_first_order` is a back-compat alias.
  - `reptile` — plain SGD adaptation on the memorize chunks, then move φ₀ toward the adapted weights; scalar = **surrogate** `Σₙ⟨φ₀ⁿ, (φ₀ⁿ−φ_Kⁿ).detach()⟩` whose grad equals the Reptile pseudo-gradient `φ₀−φ_K` (no predict term, no second-order graph). Logged `train/meta_loss` is the memorize loss for Reptile (`num_pred_pairs`=0). **Multi-GPU caveat:** the Reptile surrogate does not route through the DDP-wrapped DiT forward, so `find_unused_parameters` / manual grad all-reduce may be needed for >1 GPU; single-GPU is unaffected.
  - Launchers: [Wan2.1-T2V-1.3B-e2e-ttt.sh](examples/wanvideo/model_training/lora/Wan2.1-T2V-1.3B-e2e-ttt.sh) (MAML), [-fomaml.sh](examples/wanvideo/model_training/lora/Wan2.1-T2V-1.3B-e2e-ttt-fomaml.sh), [-reptile.sh](examples/wanvideo/model_training/lora/Wan2.1-T2V-1.3B-e2e-ttt-reptile.sh), [Wan2.2-TI2V-5B-e2e-ttt.sh](examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B-e2e-ttt.sh).
- **Sequential inference** — [Wan2.1-T2V-1.3B-e2e-ttt.py](examples/wanvideo/model_inference/Wan2.1-T2V-1.3B-e2e-ttt.py), [Wan2.2-TI2V-5B-e2e-ttt.py](examples/wanvideo/model_inference/Wan2.2-TI2V-5B-e2e-ttt.py). Set `E2E_TTT_LORA=<ckpt>` to load a meta-trained φ₀ (otherwise starts from a zero-init identity adapter).
- **Two correctness constraints** (both handled in code): (1) second-order grads (`maml` only) require double-backward attention — `enable_double_backward_attention()` disables fused flash/sage kernels and pins math SDPA; it is called only when `--e2e_algorithm maml`, since FOMAML and Reptile each do a single backward per inner step and run fine with fused kernels; (2) the differentiable override path must run **without** activation checkpointing (non-reentrant checkpoint recomputes after the param-override context exits), enforced in `compute_flow_matching_loss`.

## Where the deep docs live

The upstream docs under [docs/en/](docs/en/) are the authoritative explanation of the framework:
- [docs/en/Developer_Guide/Building_a_Pipeline.md](docs/en/Developer_Guide/Building_a_Pipeline.md) — how to add a new pipeline / model family.
- [docs/en/Developer_Guide/Integrating_Your_Model.md](docs/en/Developer_Guide/Integrating_Your_Model.md) — registering weights in `configs/model_configs.py`.
- [docs/en/Developer_Guide/Training_Diffusion_Models.md](docs/en/Developer_Guide/Training_Diffusion_Models.md) — training module conventions.
- [docs/en/Pipeline_Usage/VRAM_management.md](docs/en/Pipeline_Usage/VRAM_management.md), [docs/en/Pipeline_Usage/Accelerated_Inference.md](docs/en/Pipeline_Usage/Accelerated_Inference.md), [docs/en/Training/Offload_Training.md](docs/en/Training/Offload_Training.md) — read these before touching VRAM/offload paths.
- [docs/en/Model_Details/](docs/en/Model_Details/) — one file per supported model family with weight paths and example invocations.
