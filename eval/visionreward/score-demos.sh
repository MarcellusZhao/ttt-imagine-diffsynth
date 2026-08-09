#!/usr/bin/env bash
# Score one arm's Causal-Forcing demo clips with VisionReward-Video.
#
#   score-demos.sh <arm> [shard_index] [num_shards]
#
# Reads the layout sample-demos.sh writes: $SAMPLE_ROOT/<arm>/<prompt[:30]>/<name>.mp4.
# The full prompt is recovered by matching the truncated directory name against
# PROMPT_FILE, because these prompts run to 775 characters and cannot be filenames.
#
# Shards append to their own .jsonl under $SCORE_ROOT/<arm>/; aggregate.py merges them.
#
# Env overrides: SAMPLE_ROOT, SCORE_ROOT, PROMPT_FILE, MODEL_PATH, FRAME_SAMPLING,
# BATCH_SIZE, ENV_NAME.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROMPT_FILE="${PROMPT_FILE:-$SCRIPT_DIR/prompts/causal_forcing_demos.txt}"
SAMPLE_ROOT="${SAMPLE_ROOT:-/work/nlp/hzhao/evaluations/visionreward/causal-forcing-demos}"
SCORE_ROOT="${SCORE_ROOT:-$SAMPLE_ROOT/scores}"
MODEL_PATH="${MODEL_PATH:-/work/nlp/hzhao/checkpoints/visionreward/VisionReward-Video}"

# These clips are ~60 s but VisionReward's upstream "chat" sampling takes one frame per
# second and stops at 24 — it would score only the first 24 s and never see the drift
# the long-video arms exist to fix. "uniform" spreads the same 24 frames over the whole
# clip. Keep this identical across arms; scores from the two settings are not
# comparable.
FRAME_SAMPLING="${FRAME_SAMPLING:-uniform}"
# 1, deliberately, even though 8 measured 10.6 s/clip against 15.5 s/clip.
#
# Batching is not bitwise reproducible: bf16 matmuls dispatch different kernels at
# different batch shapes, so logits move in the last bits and a checklist item whose
# Yes/No margin is near zero can flip. Observed in practice — --verify_batching aborted
# a chunk-by-chunk shard on a disagreement at question 8 ("Does the lighting have no
# obvious errors?") while other clips verified 29/29. The answers are ±1 votes into a
# weighted mean, so a flip is a real score change, and an arm scored at a different
# batch size than the arm it is compared against carries an arbitrary bias into exactly
# the comparison this eval exists to make. ~25 extra minutes over 300 clips is a cheap
# price for every arm being scored identically.
#
# --verify_batching below stays on: at batch 1 it is a no-op, and it makes any future
# raise of this value prove itself on the job's own data before scoring anything.
BATCH_SIZE="${BATCH_SIZE:-1}"

usage() {
    echo "Usage: $(basename "$0") <arm> [shard_index] [num_shards]" >&2
    echo "Arms: base | chunk-by-chunk | chunk-by-chunk-anchored | e2e-ttt-fomaml" >&2
}

[ "$#" -ge 1 ] || { usage; exit 1; }
ARM="$1"
SHARD_INDEX="${2:-0}"
NUM_SHARDS="${3:-1}"

# RUN_NAME selects which subdirectory of SAMPLE_ROOT to score, defaulting to the arm.
# It must match the RUN_NAME the clips were sampled with — see sample-demos.sh.
RUN_DIR="${RUN_NAME:-$ARM}"

VIDEOS_PATH="$SAMPLE_ROOT/$RUN_DIR"
[ -d "$VIDEOS_PATH" ] || { echo "No sampled clips at $VIDEOS_PATH — run sample-demos.sh $ARM first." >&2; exit 1; }
[ -f "$PROMPT_FILE" ] || { echo "No prompt file at $PROMPT_FILE" >&2; exit 1; }
if [ ! -d "$MODEL_PATH" ]; then
    echo "No VisionReward weights at $MODEL_PATH — run eval/visionreward/setup.sh." >&2
    exit 1
fi

NUM_CLIPS="$(find "$VIDEOS_PATH" -mindepth 2 -maxdepth 2 -name '*.mp4' | wc -l)"
mkdir -p "$SCORE_ROOT/$RUN_DIR"
OUTPUT="$SCORE_ROOT/$RUN_DIR/shard-${SHARD_INDEX}-of-${NUM_SHARDS}.jsonl"

echo "Arm:            $ARM${RUN_NAME:+  (run: $RUN_NAME)}"
echo "Clips:          $NUM_CLIPS under $VIDEOS_PATH"
echo "Frame sampling: $FRAME_SAMPLING"
echo "Output:         $OUTPUT"
echo

python "$SCRIPT_DIR/score_videos.py" \
    --videos_path "$VIDEOS_PATH" \
    --layout custom \
    --prompt_file "$PROMPT_FILE" \
    --model_path "$MODEL_PATH" \
    --frame_sampling "$FRAME_SAMPLING" \
    --batch_size "$BATCH_SIZE" \
    --verify_batching \
    --num_shards "$NUM_SHARDS" \
    --shard_index "$SHARD_INDEX" \
    --output "$OUTPUT" \
    --skip_existing

echo
echo "Scored. Compare arms with:"
echo "  python eval/visionreward/aggregate.py \\"
echo "      base=$SCORE_ROOT/base \\"
echo "      chunk-by-chunk=$SCORE_ROOT/chunk-by-chunk \\"
echo "      chunk-by-chunk-anchored=$SCORE_ROOT/chunk-by-chunk-anchored \\"
echo "      e2e-ttt-fomaml=$SCORE_ROOT/e2e-ttt-fomaml --per_question"
