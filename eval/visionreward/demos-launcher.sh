#!/usr/bin/env bash
# Fan the 100-prompt VisionReward evaluation out over Slurm — one job per (arm, shard).
# This is the only fan-out; there are no per-arm launchers, so a new arm is a `case`
# entry in sample-demos.sh, not a new file.
#
#   demos-launcher.sh sample [arm ...]      # generate clips  (diffsynth env, 1 GPU/job)
#   demos-launcher.sh score  [arm ...]      # score them      (visionreward env)
#
# With no arms it submits all four. NUM_SHARDS splits each arm's prompt list over that
# many jobs (round-robin, all writing to the same directory with --skip_existing, so
# shards are safe to overlap or re-submit).
#
#   NUM_SHARDS=5 demos-launcher.sh sample e2e-ttt-fomaml
#   DRY_RUN=1 demos-launcher.sh sample          # print the sbatch lines only
#
# Sampling is the expensive half: 100 prompts x 24 chunks per arm. Measured at 480p on
# an h100, ~17 s/chunk for the anchored arm (50 steps at ~2.9 it/s) = ~7 min/video, so
# ~12 GPU-hours per arm; the TTT arm adds the inner loop on top. NUM_SHARDS=5 puts each
# job around 3 h, well inside sample-demos.sbatch's 2-day walltime.
#
# SAMPLE_ROOT / SCORE_ROOT / PROMPT_FILE / LORA / FRAME_SAMPLING are inherited by every
# job (sbatch exports the submitting environment), so override them here:
#   LORA=/path/to/phi_0/epoch-0.safetensors NUM_SHARDS=5 \
#       demos-launcher.sh sample e2e-ttt-fomaml
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN="${DRY_RUN:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"

ALL_ARMS=(base chunk-by-chunk chunk-by-chunk-anchored e2e-ttt-fomaml)

usage() {
    echo "Usage: $(basename "$0") <sample|score> [arm ...]" >&2
    echo "Arms (default: all four): ${ALL_ARMS[*]}" >&2
    echo "Env: NUM_SHARDS (default 1), DRY_RUN, SAMPLE_ROOT, SCORE_ROOT, LORA," >&2
    echo "     PROMPT_FILE, FRAME_SAMPLING, JOB_PREFIX" >&2
}

[ "$#" -ge 1 ] || { usage; exit 1; }
STAGE="$1"
shift

case "$STAGE" in
    sample)
        SBATCH_SCRIPT="$SCRIPT_DIR/sample-demos.sbatch"
        DEFAULT_PREFIX="vr_sample" ;;
    score)
        SBATCH_SCRIPT="$SCRIPT_DIR/score-demos.sbatch"
        DEFAULT_PREFIX="vr_score" ;;
    -h|--help|help) usage; exit 0 ;;
    *) echo "Unknown stage '$STAGE' (expected sample or score)." >&2; usage; exit 1 ;;
esac

JOB_PREFIX="${JOB_PREFIX:-$DEFAULT_PREFIX}"

if [ "$#" -gt 0 ]; then
    ARMS=("$@")
else
    ARMS=("${ALL_ARMS[@]}")
fi

for ARM in "${ARMS[@]}"; do
    case " ${ALL_ARMS[*]} " in
        *" $ARM "*) ;;
        *) echo "Unknown arm '$ARM'." >&2; usage; exit 1 ;;
    esac
done

echo "Stage:     $STAGE"
echo "Arms:      ${ARMS[*]}"
echo "Shards:    $NUM_SHARDS per arm"
echo "Jobs:      $(( ${#ARMS[@]} * NUM_SHARDS ))"
[ -n "${SAMPLE_ROOT:-}" ]     && echo "SAMPLE_ROOT:     $SAMPLE_ROOT"
[ -n "${SCORE_ROOT:-}" ]      && echo "SCORE_ROOT:      $SCORE_ROOT"
[ -n "${LORA:-}" ]            && echo "LORA:            $LORA"
[ -n "${FRAME_SAMPLING:-}" ]  && echo "FRAME_SAMPLING:  $FRAME_SAMPLING"
echo

for ARM in "${ARMS[@]}"; do
    for (( S = 0; S < NUM_SHARDS; S++ )); do
        NAME="${ARM}_${S}_${JOB_PREFIX}"
        if [ "$DRY_RUN" != "0" ]; then
            echo "sbatch -J $NAME $SBATCH_SCRIPT $ARM $S $NUM_SHARDS"
        else
            sbatch -J "$NAME" "$SBATCH_SCRIPT" "$ARM" "$S" "$NUM_SHARDS"
        fi
    done
done
