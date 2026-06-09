#!/usr/bin/env bash
# E2E-TTT VBench sampling with Wan2.2-TI2V-5B, WITHOUT inter-chunk frame conditioning
# (--no_condition_on_last_frame: chunks rely solely on the LoRA "memory scratchpad",
# with no pixel-space anchor on the previous chunk's last frame), then print the
# per-dimension evaluation commands.
#
# Model / LoRA phi_0 / algorithm / inner-loop / chunking / resolution all come from
# the YAML config; this launcher only sets the per-pass sampling knobs and forces frame
# conditioning off via --no_condition_on_last_frame (overriding the config's
# condition_on_last_frame: true). Point at a different meta-trained checkpoint by
# editing the config or overriding CONFIG/SAVE_PATH below.
#
# Layout: all clips land in a single folder named <prompt>-<index>.mp4; one folder
# covers all 16 dimensions, and temporal_flickering gets its extra indices topped up.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-$SCRIPT_DIR/configs/Wan2.2-TI2V-5B-e2e-ttt-vbench.yaml}"
SAVE_PATH="${SAVE_PATH:-/home/hzhao/ttt-imagine-diffsynth/results/vbench/Wan2.2-TI2V-5B-e2e-ttt-fomaml-without-conditioning}"

# 1) All dimensions: 1 video per prompt.
python "$SCRIPT_DIR/Wan2.2-TI2V-5B-e2e-ttt-vbench.py" \
    --config "$CONFIG" \
    --algorithm fomaml \
    --lora /home/hzhao/ttt-imagine-diffsynth/models/train/Wan2.2-TI2V-5B_e2e_ttt_fomaml-20260605-170105/epoch-0.safetensors \
    --dimension all \
    --save_path "$SAVE_PATH" \
    --num_videos_per_prompt 1 \
    --no_condition_on_last_frame \
    --skip_existing

# 2) Temporal flickering: top up to 5 videos per prompt (extra indices fill in).
python "$SCRIPT_DIR/Wan2.2-TI2V-5B-e2e-ttt-vbench.py" \
    --config "$CONFIG" \
    --algorithm fomaml \
    --lora /home/hzhao/ttt-imagine-diffsynth/models/train/Wan2.2-TI2V-5B_e2e_ttt_fomaml-20260605-170105/epoch-0.safetensors \
    --dimension temporal_flickering \
    --save_path "$SAVE_PATH" \
    --num_videos_per_prompt 5 \
    --no_condition_on_last_frame \
    --skip_existing

echo
echo "Sampling done. Evaluate each dimension with:"
for DIM in subject_consistency background_consistency temporal_flickering \
           motion_smoothness dynamic_degree aesthetic_quality imaging_quality \
           object_class multiple_objects human_action color spatial_relationship \
           scene temporal_style appearance_style overall_consistency; do
    echo "  vbench evaluate --videos_path \"$SAVE_PATH\" --dimension $DIM"
done
