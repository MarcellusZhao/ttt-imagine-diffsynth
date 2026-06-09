#!/usr/bin/env bash
# Chunk-by-chunk VBench sampling with Wan2.2-TI2V-5B, WITH inter-chunk
# conditioning (each chunk anchored on the previous chunk's last frame via native
# I2V; the duplicated anchor frame is dropped), then print the per-dimension eval commands.
#
# Model / chunking / resolution come from the YAML config; this launcher only sets
# the per-pass sampling knobs (dimension, videos-per-prompt) on the CLI. Conditioning
# is left at the config default (condition_on_last_chunk: true).
#
# Layout: all clips land in a single folder named <prompt>-<index>.mp4; one folder
# covers all 16 dimensions, and temporal_flickering gets its extra indices topped up.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-$SCRIPT_DIR/configs/Wan2.2-TI2V-5B-chunk-by-chunk-vbench.yaml}"
SAVE_PATH="${SAVE_PATH:-/home/hzhao/ttt-imagine-diffsynth/results/vbench/Wan2.2-TI2V-5B-chunk-by-chunk-with-conditioning}"

# 1) All dimensions: 1 video per prompt.
python "$SCRIPT_DIR/Wan2.2-TI2V-5B-chunk-by-chunk-vbench.py" \
    --config "$CONFIG" \
    --dimension all \
    --save_path "$SAVE_PATH" \
    --num_videos_per_prompt 1 \
    --skip_existing

# 2) Temporal flickering: top up to 5 videos per prompt (extra indices fill in).
python "$SCRIPT_DIR/Wan2.2-TI2V-5B-chunk-by-chunk-vbench.py" \
    --config "$CONFIG" \
    --dimension temporal_flickering \
    --save_path "$SAVE_PATH" \
    --num_videos_per_prompt 5 \
    --skip_existing

echo
echo "Sampling done. Evaluate each dimension with:"
for DIM in subject_consistency background_consistency temporal_flickering \
           motion_smoothness dynamic_degree aesthetic_quality imaging_quality \
           object_class multiple_objects human_action color spatial_relationship \
           scene temporal_style appearance_style overall_consistency; do
    echo "  vbench evaluate --videos_path \"$SAVE_PATH\" --dimension $DIM"
done
