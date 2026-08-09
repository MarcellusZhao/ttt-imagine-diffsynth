#!/usr/bin/env bash
# E2E-TTT VBench sampling with Wan2.2-TI2V-5B, WITH inter-chunk frame conditioning
# (generate -> memorize -> generate, scratchpad reset to phi_0 before each clip; each
# chunk additionally anchored on the previous chunk's last frame via TI2V-5B's fused
# first-frame latent), then print the eval command(s) for what was sampled.
#
# Usage: Wan2.2-TI2V-5B-e2e-ttt-fomaml-vbench.sh [dimension] [num_videos_per_prompt]
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
# and aesthetic_quality/imaging_quality reuse overall_consistency's.
#
# Model / algorithm / chunking / resolution come from the YAML config; this launcher
# sets the per-pass sampling knobs plus the phi_0 and the inner-loop/anchor settings
# that must match it. Frame conditioning is left at the defaults
# (condition_on_last_frame + condition_on_sink_frame, both on — each chunk is anchored
# on the previous chunk's last frame AND on the clip's very first frame). LORA below is
# a matching sink-trained run (ffsink); pass --no_condition_on_sink_frame if you point
# it at a checkpoint meta-trained WITHOUT the sink. Override
# CONFIG/SAVE_PATH/VBENCH_ROOT/LORA via env vars.
#
# Layout: all clips land in a single folder named <prompt>-<index>.mp4.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-$SCRIPT_DIR/../configs/Wan2.2-TI2V-5B-e2e-ttt-k3-480p-vbench.yaml}"
SAVE_PATH="${SAVE_PATH:-/work/nlp/hzhao/evaluations/vbench/Wan2.2-TI2V-5B_e2e_ttt_fomaml_k3_480p_fpc53_antidrift_inner_lr1e-5_uvl_fs_rsfps16_len_grouped_16k}"
VBENCH_ROOT="${VBENCH_ROOT:-/home/hzhao/VBench}"
LORA="${LORA:-/work/nlp/hzhao/checkpoints/e2e-ttt/Wan2.2-TI2V-5B_e2e_ttt_fomaml_k3_480p_fpc53_antidrift_inner_lr1e-5_uvl_fs_rsfps16_len_grouped_16k-20260806-203036/epoch-0.safetensors}"

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
echo "LoRA phi0: $LORA"
echo "Save path: $SAVE_PATH"
echo

python "$SCRIPT_DIR/Wan2.2-TI2V-5B-e2e-ttt-vbench.py" \
    --config "$CONFIG" \
    --vbench_root "$VBENCH_ROOT" \
    --algorithm fomaml \
    --lora "$LORA" \
    --dimension "$DIMENSION" \
    --save_path "$SAVE_PATH" \
    --num_gradient_steps 2 \
    --optimizer adamw \
    --lora_rank 128 \
    --num_videos_per_prompt "$NUM_VIDEOS" \
    --num_anchor_latent_frames 3 \
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
