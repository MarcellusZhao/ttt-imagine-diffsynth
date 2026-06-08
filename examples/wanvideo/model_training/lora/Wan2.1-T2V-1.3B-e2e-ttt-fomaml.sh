#!/usr/bin/env bash
# First-order (FOMAML) E2E-TTT meta-training for Wan2.1-T2V-1.3B. Identical to
# Wan2.1-T2V-1.3B-e2e-ttt.sh except it passes --e2e_first_order, which drops the
# Hessian term: the inner-loop graph is freed each step, fused attention is allowed
# (no math-SDPA pin), and --e2e_truncate_steps is ignored. All other knobs come from
# the same YAML config.
#   bash examples/wanvideo/model_training/lora/Wan2.1-T2V-1.3B-e2e-ttt-fomaml.sh
#   ... --config <other.yaml>            # or point at the smoke config
set -e

CONFIG="${1:-examples/wanvideo/model_training/configs/Wan2.1-T2V-1.3B-e2e-ttt-fomaml-smoke.yaml}"
# CONFIG="${1:-examples/wanvideo/model_training/configs/Wan2.1-T2V-1.3B-e2e-ttt-fomaml.yaml}"

# FOMAML's single-backward path uses far less memory than the second-order meta-backward,
# but keep expandable_segments on to avoid allocator fragmentation stranding reserved VRAM.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Invoke accelerate as a module of the ACTIVE python (conda `diffsynth` env), so it does
# not fall back to a `~/.local/bin/accelerate` tied to a different interpreter that cannot
# import `diffsynth`.
python -m accelerate.commands.launch examples/wanvideo/model_training/train_e2e_ttt.py --config "${CONFIG}" "${@:2}"
