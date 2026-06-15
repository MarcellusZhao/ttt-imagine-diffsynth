#!/usr/bin/env bash
# Sample the full VBench T2V prompt suite with Wan2.2-TI2V-5B (text-to-video
# mode), then print the per-dimension evaluation commands.
#
# Model weights (3 sharded DiT files), resolution and the umt5 tokenizer all come
# from the YAML config (configs/Wan2.2-TI2V-5B-vbench.yaml); this launcher only
# sets the per-pass sampling knobs (dimension, videos-per-prompt) on the CLI,
# which override the YAML.
#
# Layout: all clips land in a single folder named <prompt>-<index>.mp4. Because
# VBench's VBench_full_info.json selects the right prompt subset per dimension,
# one folder covers all 16 dimensions; only temporal_flickering needs its extra
# indices topped up afterwards (--skip_existing fills in the missing ones).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-$SCRIPT_DIR/configs/Wan2.2-TI2V-5B-vbench.yaml}"
SAVE_PATH="${SAVE_PATH:-/home/hzhao/ttt-imagine-diffsynth/results/vbench/Wan2.2-TI2V-5B}"

# 1) All dimensions: 1 video per prompt.
python "$SCRIPT_DIR/Wan2.2-TI2V-5B-vbench.py" \
    --config "$CONFIG" \
    --dimension all \
    --save_path "$SAVE_PATH" \
    --num_videos_per_prompt 1 \
    --skip_existing

# 2) Temporal flickering: top up to 5 videos per prompt (extra indices fill in).
python "$SCRIPT_DIR/Wan2.2-TI2V-5B-vbench.py" \
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
