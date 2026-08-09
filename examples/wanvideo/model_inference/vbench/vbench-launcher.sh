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
        ARM_PREFIX="vbench_ettt" ;;
    -h|--help|help)
        usage
        exit 0 ;;
    *)
        echo "Unknown arm '$ARM'." >&2
        usage
        exit 1 ;;
esac

JOB_PREFIX="${JOB_PREFIX:-$ARM_PREFIX}"

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
echo

for D in "${DIMENSIONS[@]}"; do
    if [ "$DRY_RUN" != "0" ]; then
        echo "sbatch -J ${D}_${JOB_PREFIX} $SBATCH_SCRIPT $D"
    else
        sbatch -J "${D}_${JOB_PREFIX}" "$SBATCH_SCRIPT" "$D"
    fi
done
