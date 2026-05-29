#!/usr/bin/env bash
# E2E-TTT meta-training for Wan2.1-T2V-1.3B. All knobs live in the YAML config;
# pass extra CLI flags after --config to override individual values.
#   bash examples/wanvideo/model_training/lora/Wan2.1-T2V-1.3B-e2e-ttt.sh
#   ... --config <other.yaml>            # or point at the smoke config
set -e

CONFIG="${1:-examples/wanvideo/model_training/configs/Wan2.1-T2V-1.3B-e2e-ttt.yaml}"

# The exact second-order meta-backward parks right at the GPU's memory ceiling, and
# the allocator strands ~1 GB as "reserved but unallocated" (fragmentation).
# expandable_segments lets CUDA grow segments instead of fragmenting, reclaiming it.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Invoke accelerate as a module of the ACTIVE python (conda `diffsynth` env), so it does
# not fall back to a `~/.local/bin/accelerate` tied to a different interpreter that cannot
# import `diffsynth`.
python -m accelerate.commands.launch examples/wanvideo/model_training/train_e2e_ttt.py --config "${CONFIG}" "${@:2}"
