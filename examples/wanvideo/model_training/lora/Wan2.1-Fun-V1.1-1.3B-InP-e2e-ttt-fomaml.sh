#!/usr/bin/env bash
# First-order (FOMAML) E2E-TTT meta-training for Wan2.1-Fun-V1.1-1.3B-InP, i2v anchor route.
#
# The algorithm comes from the CONFIG (`e2e_algorithm: fomaml`), not from a flag here — this
# launcher only chooses which YAML to run and sets the allocator env. FOMAML drops the Hessian
# term: each memorize graph is freed as its grad is taken, fused attention is allowed (no
# math-SDPA pin), and --e2e_truncate_steps is ignored.
#
#   bash examples/wanvideo/model_training/lora/Wan2.1-Fun-V1.1-1.3B-InP-e2e-ttt-fomaml.sh
#   bash .../-fomaml.sh <other.yaml>                  # first positional arg is the config
#   bash .../-fomaml.sh <cfg.yaml> --e2e_num_chunks 8 # extra args override the YAML
set -e

CONFIG="${1:-examples/wanvideo/model_training/configs/Wan2.1-Fun-V1.1-1.3B-InP-e2e-ttt-fomaml.yaml}"

# FOMAML's single-backward path uses far less memory than the second-order meta-backward,
# but keep expandable_segments on to avoid allocator fragmentation stranding reserved VRAM.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Invoke accelerate as a module of the ACTIVE python (conda `diffsynth` env), so it does
# not fall back to a `~/.local/bin/accelerate` tied to a different interpreter that cannot
# import `diffsynth`.
python -m accelerate.commands.launch examples/wanvideo/model_training/train_e2e_ttt.py --config "${CONFIG}" "${@:2}"
