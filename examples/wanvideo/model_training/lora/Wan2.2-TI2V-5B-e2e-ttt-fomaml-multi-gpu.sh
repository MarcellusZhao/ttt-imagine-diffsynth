#!/usr/bin/env bash
# Multi-GPU (DDP data-parallel) E2E-TTT meta-training for Wan2.2-TI2V-5B: one video per
# GPU per step, meta-gradients on phi_0 averaged across ranks. All training knobs live
# in the YAML config; pass extra CLI flags after --config to override individual values.
#   bash examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B-e2e-ttt-fomaml-multi-gpu.sh
#   NUM_GPUS=8 bash examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B-e2e-ttt-fomaml-multi-gpu.sh
#   ... --config <other.yaml> --use_gradient_checkpointing   # e.g. override on OOM
# NOTE: do not add --enable_model_cpu_offload here — that path skips DDP wrapping and
# is single-GPU only.
set -e

# CONFIG="${1:-examples/wanvideo/model_training/configs/Wan2.2-TI2V-5B-e2e-ttt-fomaml-multi-gpu.yaml}"
CONFIG="${1:-examples/wanvideo/model_training/configs/Wan2.2-TI2V-5B-e2e-ttt-fomaml-k3-720p.yaml}"
NUM_GPUS="${NUM_GPUS:-4}"

# Invoke accelerate as a module of the ACTIVE python (conda `diffsynth` env), so it does
# not fall back to a `~/.local/bin/accelerate` tied to a different interpreter that cannot
# import `diffsynth`. --num_processes/--multi_gpu override any `accelerate config` default,
# so this launches DDP with one process per GPU regardless of the machine-level config.
python -m accelerate.commands.launch --num_processes "${NUM_GPUS}" --multi_gpu \
    examples/wanvideo/model_training/train_e2e_ttt.py --config "${CONFIG}" "${@:2}"
