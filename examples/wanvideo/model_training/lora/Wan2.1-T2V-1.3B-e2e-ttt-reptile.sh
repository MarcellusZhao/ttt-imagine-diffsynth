#!/usr/bin/env bash
# Reptile E2E-TTT meta-training for Wan2.1-T2V-1.3B. Identical to
# Wan2.1-T2V-1.3B-e2e-ttt.sh except the YAML sets --e2e_algorithm reptile, which adapts
# the LoRA scratchpad with plain SGD on the memorize chunks and then moves phi_0 toward
# the adapted weights (pseudo-gradient phi_0 - phi_K, deposited via a surrogate so the
# stock outer loop applies it unchanged). No predict term, no second-order graph: a
# single first-order backward per inner step, fused attention allowed, --e2e_truncate_steps
# ignored. All other knobs come from the same YAML config.
#   bash examples/wanvideo/model_training/lora/Wan2.1-T2V-1.3B-e2e-ttt-reptile.sh
#   ... --config <other.yaml>            # or point at the smoke config
set -e

CONFIG="${1:-examples/wanvideo/model_training/configs/Wan2.1-T2V-1.3B-e2e-ttt-reptile.yaml}"

# Reptile's single first-order backward uses the least memory of the three variants, but
# keep expandable_segments on to avoid allocator fragmentation stranding reserved VRAM.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Invoke accelerate as a module of the ACTIVE python (conda `diffsynth` env), so it does
# not fall back to a `~/.local/bin/accelerate` tied to a different interpreter that cannot
# import `diffsynth`.
python -m accelerate.commands.launch examples/wanvideo/model_training/train_e2e_ttt.py --config "${CONFIG}" "${@:2}"
