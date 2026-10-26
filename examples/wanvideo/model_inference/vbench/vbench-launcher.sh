#!/usr/bin/env bash
# Submit one Slurm job per VBench dimension, for ANY of the four Wan2.2-TI2V-5B eval
# arms. This is the fan-out form of the per-arm `*.sbatch` scripts, each of which
# samples ONE dimension per GPU. All jobs of a submission write to the SAME SAVE_PATH:
# the evaluator wants a single folder covering every dimension and recovers the
# dimension from the filename, not the path.
#
# Arms (first positional arg):
#   base                     plain text-to-video, one shot, no chunking       (no LoRA)
#   chunk-by-chunk           chunked, previous chunk's last frame via I2V     (no LoRA)
#   chunk-by-chunk-anchored  chunked, full E2E-TTT anchoring (wide k + sink)  (no LoRA)
#   e2e-ttt-fomaml           chunked, full anchoring + LoRA scratchpad + TTT  (needs LORA)
#
# The 11 dimensions below are the per-dimension prompt FILES, which cover all 16 eval
# dimensions: background_consistency is scored on scene's clips,
# dynamic_degree/motion_smoothness on subject_consistency's, and
# aesthetic_quality/imaging_quality on overall_consistency's.
#
# Usage:
#   vbench-launcher.sh <arm>                      # all 11 dimensions
#   vbench-launcher.sh <arm> scene color          # just these
#   DRY_RUN=1 vbench-launcher.sh <arm>            # print the sbatch lines only
#
# temporal_flickering is SEED-SHARDED on the arms whose launcher supports it (today just
# e2e-ttt-fomaml, via its SEED_INDEX env var). That dimension is 75 prompts x 5 clips =
# 375 against every other dimension's ~85, so on one GPU it is ~65 h vs ~13 h at 480p and
# it has hit the slurm wall three times. The 5 clips cannot be dropped -- VBench's static
# filter runs for this dimension only and keeps up to 5 *static* split-clips per prompt
# out of that prompt's pool, so a thinner pool biases the sample -- so it is split into
# TF_SEED_SHARDS jobs (default 5), each running all 75 prompts at ONE clip index. Seeds
# are unchanged, so the result is byte-identical to the single-job form. Sharding by seed
# rather than by prompt range is deliberate: a dead seed job thins every prompt's pool
# evenly, whereas a dead prompt-range job leaves its prompts with no clips at all and the
# dimension is silently scored on a smaller prompt subset.
#   TF_SEED_SHARDS=1 vbench-launcher.sh e2e-ttt-fomaml temporal_flickering   # old single job
# Arms without SEED_INDEX support always get the single-job form.
#
# NUM_SHARDS splits every OTHER dimension by prompt range instead: NUM_SHARDS=8 submits
# 8 jobs per dimension, each taking a contiguous ~1/8 of that dimension's prompts, all
# writing to the same SAVE_PATH. At ~44 min per 60 s clip one dimension is 52-68 h on a
# single GPU -- close to the 3-day wall and hopeless against a deadline -- while eight
# shards are 7-9 h each. Pair it with TIME_LIMIT (e.g. TIME_LIMIT=10:00:00), which is
# applied to `sbatch` itself and overrides the .sbatch's 3-day #SBATCH --time: a job
# that asks for 3 days will not backfill into the gaps on a full cluster, so the shards
# would queue behind everything. Only arms whose .sh reads NUM_SHARDS/SHARD_INDEX
# accept this (today just e2e-ttt-fomaml); the rest refuse rather than silently
# sampling the whole dimension N times.
#   NUM_SHARDS=8 TIME_LIMIT=10:00:00 vbench-launcher.sh e2e-ttt-fomaml scene
#
# CONFIG / SAVE_PATH / LORA / VBENCH_ROOT are read by the per-arm launcher inside the
# job and are inherited from this shell (sbatch exports the submitting environment by
# default), so override them here and every job in the fan-out picks them up:
#   SAVE_PATH=/work/nlp/hzhao/evaluations/vbench/k3-480p \
#   CONFIG=examples/wanvideo/model_inference/configs/Wan2.2-TI2V-5B-e2e-ttt-k3-480p-vbench.yaml \
#   LORA=/path/to/phi_0/epoch-0.safetensors \
#       vbench-launcher.sh e2e-ttt-fomaml
#
# JOB_PREFIX (default: per-arm, see ARM_PREFIX below) names the jobs
# `<dimension>_<JOB_PREFIX>`, which is also the log filename via the sbatch script's %x.
# Bump it per experiment arm so concurrent arms don't interleave in /home/hzhao/logs.
#
# Do NOT also submit the `all` dimension alongside this: temporal_flickering is sampled
# at 5 clips/prompt here and `all` would sample the same prompts at 1, so the two
# collide on <prompt>-0.mp4 (--skip_existing then hides which arm produced it).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN="${DRY_RUN:-0}"

ALL_DIMENSIONS=(
    subject_consistency scene overall_consistency object_class multiple_objects
    human_action color spatial_relationship temporal_style appearance_style
    temporal_flickering
)

usage() {
    echo "Usage: $(basename "$0") <arm> [dimension ...]" >&2
    echo "Arms:" >&2
    echo "  base                     plain text-to-video, one shot" >&2
    echo "  chunk-by-chunk           chunked, last-frame I2V conditioning" >&2
    echo "  chunk-by-chunk-anchored  chunked, full E2E-TTT anchoring, no LoRA" >&2
    echo "  e2e-ttt-fomaml           chunked, full anchoring + LoRA scratchpad + TTT" >&2
    echo "Dimensions (default: all 11):" >&2
    echo "  ${ALL_DIMENSIONS[*]}" >&2
}

if [ "$#" -lt 1 ]; then
    usage
    exit 1
fi

ARM="$1"
shift

# Each arm is just a different .sbatch + default job-name suffix; everything else
# (dimension handling, env overrides, eval printout) lives in the per-arm launcher.
case "$ARM" in
    base)
        SBATCH_SCRIPT="$SCRIPT_DIR/Wan2.2-TI2V-5B-vbench.sbatch"
        ARM_PREFIX="vbench_base" ;;
    chunk-by-chunk)
        SBATCH_SCRIPT="$SCRIPT_DIR/Wan2.2-TI2V-5B-chunk-by-chunk-vbench.sbatch"
        ARM_PREFIX="vbench_cbc" ;;
    chunk-by-chunk-anchored)
        SBATCH_SCRIPT="$SCRIPT_DIR/Wan2.2-TI2V-5B-chunk-by-chunk-anchored-vbench.sbatch"
        ARM_PREFIX="vbench_cbc_anch" ;;
    e2e-ttt-fomaml)
        SBATCH_SCRIPT="$SCRIPT_DIR/Wan2.2-TI2V-5B-e2e-ttt-fomaml-vbench.sbatch"
        ARM_PREFIX="vbench_ettt"
        # This arm's launcher understands NUM_SHARDS/SHARD_INDEX (prompt-range shards).
        ARM_PROMPT_SHARDING=1
        # This arm's launcher honours SEED_INDEX. Setting it for an arm that does not
        # would be silently ignored and submit N jobs all doing the SAME full 5-clip
        # work, so the fan-out below is gated on this flag. Add it here when another
        # arm gains a per-seed driver.
        ARM_SEED_SHARDING=1 ;;
    -h|--help|help)
        usage
        exit 0 ;;
    *)
        echo "Unknown arm '$ARM'." >&2
        usage
        exit 1 ;;
esac

JOB_PREFIX="${JOB_PREFIX:-$ARM_PREFIX}"
ARM_SEED_SHARDING="${ARM_SEED_SHARDING:-0}"
# Clip indices to cover for temporal_flickering. 5 matches VBench's protocol; 1 (or 0)
# restores the single all-indices-in-one-job form.
TF_SEED_SHARDS="${TF_SEED_SHARDS:-5}"
ARM_PROMPT_SHARDING="${ARM_PROMPT_SHARDING:-0}"
# Prompt-range shards per dimension (see NUM_SHARDS in the header). 1 = one job per
# dimension, the old behaviour.
NUM_SHARDS="${NUM_SHARDS:-1}"
if [ "$NUM_SHARDS" -gt 1 ] && [ "$ARM_PROMPT_SHARDING" != "1" ]; then
    echo "Arm '$ARM' does not support NUM_SHARDS: its launcher would ignore SHARD_INDEX" >&2
    echo "and every job would sample the WHOLE dimension. Add the sharding block to its" >&2
    echo ".sh first (see Wan2.2-TI2V-5B-e2e-ttt-fomaml-vbench.sh)." >&2
    exit 1
fi
# Wall-clock override applied to `sbatch` itself, so it wins over the .sbatch's #SBATCH
# --time. Sharded jobs are hours, not days, and asking for 3 days makes them invisible
# to the backfill scheduler on a full cluster — which is the whole point of sharding.
TIME_LIMIT="${TIME_LIMIT:-}"
# SBATCH_EXTRA (env): raw sbatch options appended to every submission, for cluster-specific
# resource blocks the arm .sbatch files do not know about, e.g. on lyra:
#   SBATCH_EXTRA="--partition=b200 --cpus-per-task=12 --mem=180G" CONDA_ENV=diffsynth-b200 …
# (CONDA_ENV is inherited by the job like SAVE_PATH/LORA and picked up by the arm .sbatch.)
SBATCH_EXTRA="${SBATCH_EXTRA:-}"

# Positional args select a subset; no args means the full sweep.
if [ "$#" -gt 0 ]; then
    DIMENSIONS=("$@")
else
    DIMENSIONS=("${ALL_DIMENSIONS[@]}")
fi

echo "Arm:       $ARM"
echo "Submitting ${#DIMENSIONS[@]} job(s) from $SBATCH_SCRIPT"
echo "Job names: <dimension>_${JOB_PREFIX}"
[ -n "${SAVE_PATH:-}" ] && echo "SAVE_PATH: $SAVE_PATH (inherited by every job)"
[ -n "${CONFIG:-}" ]    && echo "CONFIG:    $CONFIG"
[ -n "${LORA:-}" ]      && echo "LORA:      $LORA"
# e2e-ttt-fomaml only: 1 = memorize on the empty prompt, 0 = on the chunk's prompt. Must
# match the LORA's train/drop_text_in_memorize; the arm's .sh defaults it to 1.
[ -n "${DROP_TEXT_IN_MEMORIZE:-}" ] && echo "DROP_TEXT_IN_MEMORIZE: $DROP_TEXT_IN_MEMORIZE (inherited by every job)"
[ -n "${EXTRA_ARGS:-}" ] && echo "EXTRA_ARGS: $EXTRA_ARGS (inherited by every job; appended last, overrides the config)"
[ -n "$SBATCH_EXTRA" ] && echo "SBATCH_EXTRA: $SBATCH_EXTRA (added to every sbatch call)"
[ -n "${CONDA_ENV:-}" ] && echo "CONDA_ENV: $CONDA_ENV (inherited by every job)"
echo

# submit <job-name> <dimension> [seed-index]
# The env-assignment prefix is applied to `sbatch` ITSELF, not to this function: sbatch
# forwards the submitting environment to the job, so SEED_INDEX has to be in sbatch's
# environment, and `VAR=v some_function` does not reliably put it there.
submit() {
    local NAME="$1" DIM="$2" SEED="${3:-}" SHARD="${4:-}"
    local TIME_ARGS=()
    [ -n "$TIME_LIMIT" ] && TIME_ARGS=(--time="$TIME_LIMIT")
    # shellcheck disable=SC2206  # word-splitting SBATCH_EXTRA is the point
    [ -n "$SBATCH_EXTRA" ] && TIME_ARGS+=($SBATCH_EXTRA)
    if [ "$DRY_RUN" != "0" ]; then
        echo "${SEED:+SEED_INDEX=$SEED }${SHARD:+NUM_SHARDS=$NUM_SHARDS SHARD_INDEX=$SHARD }sbatch ${TIME_ARGS[*]:-} -J $NAME $SBATCH_SCRIPT $DIM"
    elif [ -n "$SEED" ]; then
        SEED_INDEX="$SEED" sbatch ${TIME_ARGS[@]+"${TIME_ARGS[@]}"} -J "$NAME" "$SBATCH_SCRIPT" "$DIM"
    elif [ -n "$SHARD" ]; then
        NUM_SHARDS="$NUM_SHARDS" SHARD_INDEX="$SHARD" \
            sbatch ${TIME_ARGS[@]+"${TIME_ARGS[@]}"} -J "$NAME" "$SBATCH_SCRIPT" "$DIM"
    else
        sbatch ${TIME_ARGS[@]+"${TIME_ARGS[@]}"} -J "$NAME" "$SBATCH_SCRIPT" "$DIM"
    fi
}

for D in "${DIMENSIONS[@]}"; do
    if [ "$D" = "temporal_flickering" ] && [ "$ARM_SEED_SHARDING" = "1" ] \
       && [ "$TF_SEED_SHARDS" -gt 1 ]; then
        echo "  $D -> $TF_SEED_SHARDS seed shards (clip indices 0..$((TF_SEED_SHARDS - 1)))"
        for I in $(seq 0 $((TF_SEED_SHARDS - 1))); do
            submit "${D}_s${I}_${JOB_PREFIX}" "$D" "$I"
        done
    elif [ "$NUM_SHARDS" -gt 1 ]; then
        echo "  $D -> $NUM_SHARDS prompt shards"
        for I in $(seq 0 $((NUM_SHARDS - 1))); do
            submit "${D}_p${I}_${JOB_PREFIX}" "$D" "" "$I"
        done
    else
        submit "${D}_${JOB_PREFIX}" "$D"
    fi
done
