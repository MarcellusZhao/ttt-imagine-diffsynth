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
# SEED_INDEX (env, temporal_flickering only) — seed-shard that one dimension.
#   temporal_flickering is 75 prompts x 5 clips = 375 clips against every other
#   dimension's ~85, i.e. ~65 h vs ~13 h at 480p on one GPU, and it has hit the slurm
#   wall three times. The 5 clips are NOT redundant (VBench's static filter runs for
#   this dimension only and keeps up to 5 *static* split-clips per prompt out of that
#   prompt's pool, so a thinner pool biases the sample), so the work is split across
#   jobs instead: SEED_INDEX=i runs ALL 75 prompts at clip index i only, via
#   Wan2.2-TI2V-5B-e2e-ttt-temporal-flickering-vbench.py. Five such jobs cover the
#   dimension in ~13 h. Seeds are unchanged — index i still means
#   <prompt>-<i>.mp4 with chunk seeds base_seed + i*num_chunks + k — so a sharded run
#   is byte-identical to the all-in-one one. Unset SEED_INDEX keeps the old single-job
#   behaviour. vbench-launcher.sh fans these out automatically for this arm.
#
# NUM_SHARDS / SHARD_INDEX (env) — split ONE dimension's PROMPTS across NUM_SHARDS
#   jobs, contiguous and balanced; this job runs shard SHARD_INDEX (0-based). All
#   shards may share a SAVE_PATH. Mutually exclusive with SEED_INDEX. See the block
#   below. vbench-launcher.sh fans these out.
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
CONFIG="${CONFIG:-$SCRIPT_DIR/../configs/Wan2.2-TI2V-5B-e2e-ttt-k3-720p-fpc133-24fps-vbench.yaml}"
SAVE_PATH="${SAVE_PATH:-/work/nlp/hzhao/evaluations/vbench/Wan2.2-TI2V-5B_e2e_ttt_fomaml_k3_720p_fpc133_24fps_antidrift_buf96_incremental_data_v3_16k_24fps_60s}"
VBENCH_ROOT="${VBENCH_ROOT:-/home/hzhao/VBench}"
LORA="${LORA:-/work/nlp/hzhao/checkpoints/e2e-ttt/Wan2.2-TI2V-5B_e2e_ttt_fomaml_k3_720p_fpc133_24fps_antidrift_buf96_incremental_data_v3_16k-20260827-003600/step-502.safetensors}"

DIMENSION="${1:-all}"
# temporal_flickering needs 5 clips per prompt; every other dimension gets 1.
if [ "$DIMENSION" = "temporal_flickering" ]; then
    NUM_VIDEOS="${2:-5}"
else
    NUM_VIDEOS="${2:-1}"
fi

# Seed shard (see the SEED_INDEX note in the header). Only temporal_flickering has a
# per-seed driver; anywhere else it would silently do nothing, so refuse instead.
SEED_INDEX="${SEED_INDEX:-}"
if [ -n "$SEED_INDEX" ]; then
    if [ "$DIMENSION" != "temporal_flickering" ]; then
        echo "SEED_INDEX is only supported for --dimension temporal_flickering (got '$DIMENSION')." >&2
        echo "Every other dimension is 1 clip/prompt and needs no seed sharding." >&2
        exit 1
    fi
    case "$SEED_INDEX" in
        ''|*[!0-9]*)
            echo "SEED_INDEX must be a non-negative integer, got '$SEED_INDEX'." >&2
            exit 1 ;;
    esac
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

# ── Prompt-range sharding (NUM_SHARDS / SHARD_INDEX) ─────────────────────────
# Split ONE dimension's prompt list across NUM_SHARDS jobs, this one taking shard
# SHARD_INDEX (0-based). Ranges are contiguous, disjoint and balanced to within one
# prompt, so the shards together cover the dimension exactly once and all of them can
# share a SAVE_PATH (clips are named <prompt>-<index>.mp4, and the prompt sets do not
# overlap).
#
# This is the lever for a deadline: at ~44 min per 60 s clip a whole dimension is
# 52-68 h on one GPU, which is both close to the 3-day wall and unschedulable when the
# cluster is full. Eight shards is 9-12 clips each, 7-9 h, which backfills.
#
# Sharding by prompt RANGE is right here and wrong for temporal_flickering: at 1 clip
# per prompt a dead range job leaves those prompts with NO clips at all, whereas that
# dimension's 5-clip pool can be split by seed so a dead job only thins each prompt's
# pool. Hence SEED_INDEX and these are mutually exclusive. A dead range job is
# recoverable by resubmitting it — --skip_existing is on, so it resumes.
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_FLAGS=()
if [ "$NUM_SHARDS" -gt 1 ]; then
    if [ -n "$SEED_INDEX" ]; then
        echo "NUM_SHARDS and SEED_INDEX are two different ways to split one dimension;" >&2
        echo "set only one of them." >&2
        exit 1
    fi
    if [ "$SHARD_INDEX" -lt 0 ] || [ "$SHARD_INDEX" -ge "$NUM_SHARDS" ]; then
        echo "SHARD_INDEX must be in 0..$((NUM_SHARDS - 1)), got '$SHARD_INDEX'." >&2
        exit 1
    fi
    if [ "$DIMENSION" = "all" ]; then
        PROMPT_FILE="$VBENCH_ROOT/prompts/all_dimension.txt"
    else
        PROMPT_FILE="$VBENCH_ROOT/prompts/prompts_per_dimension/$DIMENSION.txt"
    fi
    # Must count prompts the way the driver's read_prompts() does: strip, drop blanks.
    NUM_PROMPTS="$(grep -c '[^[:space:]]' "$PROMPT_FILE")"
    BASE=$((NUM_PROMPTS / NUM_SHARDS))
    REM=$((NUM_PROMPTS % NUM_SHARDS))
    EXTRA=$((SHARD_INDEX < REM ? SHARD_INDEX : REM))
    START=$((SHARD_INDEX * BASE + EXTRA))
    SIZE=$((BASE + (SHARD_INDEX < REM ? 1 : 0)))
    END=$((START + SIZE))
    if [ "$SIZE" -eq 0 ]; then
        echo "Shard $SHARD_INDEX of $NUM_SHARDS is empty ($NUM_PROMPTS prompts); nothing to do."
        exit 0
    fi
    SHARD_FLAGS=(--start_index "$START" --end_index "$END")
fi

if [ -n "$SEED_INDEX" ]; then
    echo "Dimension: $DIMENSION | seed shard: clip index $SEED_INDEX only (all prompts)"
elif [ "$NUM_SHARDS" -gt 1 ]; then
    echo "Dimension: $DIMENSION | prompt shard $SHARD_INDEX/$NUM_SHARDS: prompts [$START,$END) of $NUM_PROMPTS | $NUM_VIDEOS clip(s) per prompt"
else
    echo "Dimension: $DIMENSION | $NUM_VIDEOS clip(s) per prompt"
fi
# DROP_TEXT_IN_MEMORIZE (env): 1 -> --drop_text_in_memorize (memorize on the EMPTY prompt),
# 0 -> --no_drop_text_in_memorize (memorize on the chunk's own prompt). See COMMON_FLAGS.
DROP_TEXT_IN_MEMORIZE="${DROP_TEXT_IN_MEMORIZE:-1}"
case "$DROP_TEXT_IN_MEMORIZE" in
    1) DROP_TEXT_FLAG="--drop_text_in_memorize" ;;
    0) DROP_TEXT_FLAG="--no_drop_text_in_memorize" ;;
    *) echo "DROP_TEXT_IN_MEMORIZE must be 0 or 1, got '$DROP_TEXT_IN_MEMORIZE'." >&2; exit 1 ;;
esac

echo "Config:    $CONFIG"
echo "LoRA phi0: $LORA"
echo "Save path: $SAVE_PATH"
echo "Text in memorize loss: $([ "$DROP_TEXT_IN_MEMORIZE" = 1 ] && echo 'DROPPED (empty prompt)' || echo 'KEPT (chunk prompt)') -> $DROP_TEXT_FLAG"
# EXTRA_ARGS: extra driver flags appended LAST, so they override the YAML config and the
# flags above (argparse keeps the last occurrence). Used for one-variable controls that must
# show up in the job log, e.g. EXTRA_ARGS="--inner_lr_init 0" samples phi_0 STATIC (AdamW
# step is exactly 0, same code path and same chunk seeds as the TTT arm). Inherited from the
# submitting environment through vbench-launcher.sh like SAVE_PATH / LORA.
EXTRA_ARGS="${EXTRA_ARGS:-}"
if [ -n "$EXTRA_ARGS" ]; then echo "Extra args: $EXTRA_ARGS (appended last, override config)"; fi
echo

# ONE flag list for both entrypoints. These are the settings that must match the phi_0
# checkpoint, so they must not be able to differ between the sharded and unsharded paths.
COMMON_FLAGS=(
    --config "$CONFIG"
    --vbench_root "$VBENCH_ROOT"
    --algorithm fomaml
    --lora "$LORA"
    --save_path "$SAVE_PATH"
    --num_gradient_steps 2
    --optimizer adamw
    --lora_rank 128
    --num_anchor_latent_frames 3
    # Text in the memorize loss is part of the inner update rule and MUST match how $LORA
    # was meta-trained (wandb: train/drop_text_in_memorize == 1 -> DROP_TEXT_IN_MEMORIZE=1).
    # The flag is passed EXPLICITLY in both directions because the YAML CONFIG also sets
    # `drop_text_in_memorize` as an argparse default, so omitting the flag does not mean
    # "off" -- it means "whatever the YAML says". A mismatch is silent either way: the
    # step-1844 uvl16k phi_0 (trained text-ON) was sampled text-OFF for a week before
    # anyone noticed. Default 1 = the text-free phi_0s every 24fps 720p arm has used.
    "$DROP_TEXT_FLAG"
    --skip_existing
)

if [ -n "$SEED_INDEX" ]; then
    # Seed-sharded temporal_flickering: all 75 prompts, this one clip index. The driver
    # forces --dimension temporal_flickering and ignores --num_videos_per_prompt.
    python "$SCRIPT_DIR/Wan2.2-TI2V-5B-e2e-ttt-temporal-flickering-vbench.py" \
        "${COMMON_FLAGS[@]}" \
        --seed_index "$SEED_INDEX" \
        ${EXTRA_ARGS:+$EXTRA_ARGS}
else
    python "$SCRIPT_DIR/Wan2.2-TI2V-5B-e2e-ttt-vbench.py" \
        "${COMMON_FLAGS[@]}" \
        ${SHARD_FLAGS[@]+"${SHARD_FLAGS[@]}"} \
        --dimension "$DIMENSION" \
        --num_videos_per_prompt "$NUM_VIDEOS" \
        ${EXTRA_ARGS:+$EXTRA_ARGS}
fi

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
