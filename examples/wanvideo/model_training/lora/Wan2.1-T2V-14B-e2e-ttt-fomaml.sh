#!/usr/bin/env bash
# First-order (FOMAML) E2E-TTT meta-training for Wan2.1-T2V-14B. Identical to the
# 1.3B/5B fomaml launchers except it points at the 14B config. --e2e_first_order
# (set in the YAML) drops the Hessian term: the inner-loop graph is freed each step,
# fused attention is allowed (no math-SDPA pin), and --e2e_truncate_steps is ignored.
# All other knobs come from the YAML config.
#   bash examples/wanvideo/model_training/lora/Wan2.1-T2V-14B-e2e-ttt-fomaml.sh
#   ... --config <other.yaml> --use_gradient_checkpointing   # e.g. override on OOM
set -e

CONFIG="${1:-examples/wanvideo/model_training/configs/Wan2.1-T2V-14B-e2e-ttt-fomaml.yaml}"

# FOMAML's single-backward path uses far less memory than the second-order meta-backward,
# but the 14B DiT (~28 GB) leaves little slack — keep expandable_segments on to avoid
# allocator fragmentation stranding reserved VRAM.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Invoke accelerate as a module of the ACTIVE python (conda `diffsynth` env), so it does
# not fall back to a `~/.local/bin/accelerate` tied to a different interpreter that cannot
# import `diffsynth`.
python -m accelerate.commands.launch examples/wanvideo/model_training/train_e2e_ttt.py --config "${CONFIG}" "${@:2}"
