#!/usr/bin/env bash
# E2E-TTT meta-training for Wan2.2-TI2V-5B. All knobs live in the YAML config;
# pass extra CLI flags after --config to override individual values.
#   bash examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B-e2e-ttt-fomaml.sh
#   ... --config <other.yaml> --use_gradient_checkpointing   # e.g. override on OOM
set -e

CONFIG="${1:-examples/wanvideo/model_training/configs/Wan2.2-TI2V-5B-e2e-ttt-fomaml.yaml}"

# Invoke accelerate as a module of the ACTIVE python (conda `diffsynth` env), so it does
# not fall back to a `~/.local/bin/accelerate` tied to a different interpreter that cannot
# import `diffsynth`.
python -m accelerate.commands.launch examples/wanvideo/model_training/train_e2e_ttt.py --config "${CONFIG}" "${@:2}"
