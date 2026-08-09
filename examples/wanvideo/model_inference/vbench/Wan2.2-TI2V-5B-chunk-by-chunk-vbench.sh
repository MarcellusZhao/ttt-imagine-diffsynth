#!/usr/bin/env bash
# NO-ADAPTATION BASELINE: chunk-by-chunk VBench sampling with Wan2.2-TI2V-5B, WITH
# inter-chunk conditioning (each chunk anchored on the previous chunk's last frame via
# native I2V; the duplicated anchor frame is dropped) — base model, no LoRA, no
# test-time training — then print the eval command(s) for what was sampled.
#
# Usage: Wan2.2-TI2V-5B-chunk-by-chunk-vbench.sh [dimension] [num_videos_per_prompt]
#
#   dimension              a name under $VBENCH_ROOT/prompts/prompts_per_dimension/
#                          (e.g. human_action), or `all` for prompts/all_dimension.txt.
#                          Default: all.
#   num_videos_per_prompt  clips per prompt. Default: 1, or 5 for temporal_flickering
#                          (VBench's static filter discards most of that dimension's
#                          clips, so it needs extra indices).
#
# One dimension per process is the unit of parallelism: run one invocation per GPU, all
# writing to the SAME SAVE_PATH — the evaluator wants one folder covering every
# dimension and recovers the dimension from the prompt, not from the path. The 11
# per-dimension prompt files cover all 16 eval dimensions: background_consistency
# reuses scene's prompts, dynamic_degree/motion_smoothness reuse subject_consistency's,
# and aesthetic_quality/imaging_quality reuse overall_consistency's. Use
# vbench-launcher.sh to fan the 11 out over Slurm.
#
# Model / chunking / resolution come from the YAML config; this launcher only sets the
# per-pass sampling knobs on the CLI. Conditioning is left at the config default
# (condition_on_last_chunk: true). Override CONFIG/SAVE_PATH/VBENCH_ROOT via env vars.
#
# Layout: all clips land in a single folder named <prompt>-<index>.mp4.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-$SCRIPT_DIR/../configs/Wan2.2-TI2V-5B-chunk-by-chunk-vbench.yaml}"
SAVE_PATH="${SAVE_PATH:-/work/nlp/hzhao/evaluations/vbench/Wan2.2-TI2V-5B-chunk-by-chunk-480h-832w-60s}"
VBENCH_ROOT="${VBENCH_ROOT:-/home/hzhao/VBench}"

DIMENSION="${1:-all}"
# temporal_flickering needs 5 clips per prompt; every other dimension gets 1.
if [ "$DIMENSION" = "temporal_flickering" ]; then
    NUM_VIDEOS="${2:-5}"
else
    NUM_VIDEOS="${2:-1}"
fi

# Fail fast on a typo'd dimension rather than after the model has loaded.
if [ "$DIMENSION" != "all" ] && \
   [ ! -f "$VBENCH_ROOT/prompts/prompts_per_dimension/$DIMENSION.txt" ]; then
    echo "Unknown dimension '$DIMENSION'. Available:" >&2
    for F in "$VBENCH_ROOT"/prompts/prompts_per_dimension/*.txt; do
        echo "  $(basename "$F" .txt)" >&2
    done
    echo "  all   (prompts/all_dimension.txt — the union of the above)" >&2
    exit 1
fi

echo "Dimension: $DIMENSION | $NUM_VIDEOS clip(s) per prompt"
echo "Config:    $CONFIG"
echo "Save path: $SAVE_PATH"
echo

python "$SCRIPT_DIR/Wan2.2-TI2V-5B-chunk-by-chunk-vbench.py" \
    --config "$CONFIG" \
    --vbench_root "$VBENCH_ROOT" \
    --dimension "$DIMENSION" \
    --save_path "$SAVE_PATH" \
    --num_videos_per_prompt "$NUM_VIDEOS" \
    --skip_existing

echo
if [ "$DIMENSION" = "all" ]; then
    echo "Sampling done. Evaluate each dimension with:"
    for DIM in subject_consistency background_consistency temporal_flickering \
               motion_smoothness dynamic_degree aesthetic_quality imaging_quality \
               object_class multiple_objects human_action color spatial_relationship \
               scene temporal_style appearance_style overall_consistency; do
        echo "  vbench evaluate --videos_path \"$SAVE_PATH\" --dimension $DIM"
    done
else
    echo "Sampling done. Evaluate with:"
    echo "  vbench evaluate --videos_path \"$SAVE_PATH\" --dimension $DIMENSION"
    # The five quality dimensions have no prompt file of their own — they are scored on
    # another dimension's clips, so sampling that dimension also unlocks them.
    case "$DIMENSION" in
        subject_consistency)
            echo "  vbench evaluate --videos_path \"$SAVE_PATH\" --dimension dynamic_degree"
            echo "  vbench evaluate --videos_path \"$SAVE_PATH\" --dimension motion_smoothness" ;;
        scene)
            echo "  vbench evaluate --videos_path \"$SAVE_PATH\" --dimension background_consistency" ;;
        overall_consistency)
            echo "  vbench evaluate --videos_path \"$SAVE_PATH\" --dimension aesthetic_quality"
            echo "  vbench evaluate --videos_path \"$SAVE_PATH\" --dimension imaging_quality" ;;
    esac
fi
