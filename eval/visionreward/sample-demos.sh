#!/usr/bin/env bash
# Sample the 100 Causal-Forcing demo prompts with one Wan2.2-TI2V-5B arm, in the layout
# score-demos.sh expects.
#
#   sample-demos.sh <arm> [shard_index] [num_shards]
#
# Arms (same four as examples/wanvideo/model_inference/vbench/vbench-launcher.sh):
#   base                     plain text-to-video, one shot                    (no LoRA)
#   chunk-by-chunk           chunked, previous chunk's last frame via I2V     (no LoRA)
#   chunk-by-chunk-anchored  chunked, full E2E-TTT anchoring (k=3 + sink)     (no LoRA)
#   e2e-ttt-fomaml           chunked, full anchoring + LoRA scratchpad + TTT  (needs LORA)
#
# Geometry is pinned to the currently-active k3/480p experiment (the one uncommented in
# custom-prompts/Wan2.2-TI2V-5B-custom-prompt-eval.sh): 480x832, 24 chunks, fpc=53 with
# k=3 + first-frame sink for the two anchored arms, fpc=41 for plain chunk-by-chunk.
# VisionReward scores are length-sensitive, so keep NUM_CHUNKS / resolution identical
# across arms — the anchored arms land on 53 + 23*40 = 973 frames and `base` is set to
# exactly that; plain chunk-by-chunk can only reach 41 + 23*40 = 961 at integer fpc,
# 12 frames (0.75 s) short. Changing an arm's fpc or k here means changing the matching
# training config too.
#
# Sharding: the driver has no --start_index, so a shard is materialised as its own
# prompt file (round-robin over the 100 prompts) and passed as --prompt_file. Shards
# share SAMPLE_ROOT and use --skip_existing, so overlapping or re-submitted shards are
# safe and resume rather than redo.
#
# Env overrides: SAMPLE_ROOT, PROMPT_FILE, LORA, NUM_CHUNKS, HEIGHT, WIDTH, SEED.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DRIVERS="$REPO_ROOT/examples/wanvideo/model_inference/custom-prompts"

PROMPT_FILE="${PROMPT_FILE:-$SCRIPT_DIR/prompts/causal_forcing_demos.txt}"
SAMPLE_ROOT="${SAMPLE_ROOT:-/work/nlp/hzhao/evaluations/visionreward/causal-forcing-demos}"
LORA="${LORA:-/work/nlp/hzhao/checkpoints/e2e-ttt/Wan2.2-TI2V-5B_e2e_ttt_fomaml_k3_480p_fpc53_antidrift_uvl_fs_rsfps16_len_grouped_16k-20260804-200034/epoch-0.safetensors}"

NUM_CHUNKS="${NUM_CHUNKS:-24}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
SEED="${SEED:-0}"
# 53 + (NUM_CHUNKS-1)*40 — the total the two anchored arms produce, so `base` matches.
BASE_NUM_FRAMES="${BASE_NUM_FRAMES:-$(( 53 + (NUM_CHUNKS - 1) * 40 ))}"

NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走}"

usage() {
    echo "Usage: $(basename "$0") <arm> [shard_index] [num_shards]" >&2
    echo "Arms: base | chunk-by-chunk | chunk-by-chunk-anchored | e2e-ttt-fomaml" >&2
}

[ "$#" -ge 1 ] || { usage; exit 1; }
ARM="$1"
SHARD_INDEX="${2:-0}"
NUM_SHARDS="${3:-1}"

case "$ARM" in
    base|chunk-by-chunk|chunk-by-chunk-anchored|e2e-ttt-fomaml) ;;
    -h|--help|help) usage; exit 0 ;;
    *) echo "Unknown arm '$ARM'." >&2; usage; exit 1 ;;
esac

# Fail before ~10 GB of weights load, not after.
[ -f "$PROMPT_FILE" ] || { echo "No prompt file at $PROMPT_FILE" >&2; exit 1; }
if [ "$SHARD_INDEX" -ge "$NUM_SHARDS" ] || [ "$SHARD_INDEX" -lt 0 ]; then
    echo "shard_index $SHARD_INDEX outside [0, $NUM_SHARDS)" >&2
    exit 1
fi
if [ "$ARM" = "e2e-ttt-fomaml" ] && [ ! -f "$LORA" ]; then
    echo "LORA checkpoint not found: $LORA" >&2
    echo "A missing path would silently fall back to a zero-init identity adapter," >&2
    echo "turning this arm into the anchored baseline. Set LORA=... explicitly." >&2
    exit 1
fi

# RUN_NAME names the output subdirectory, defaulting to the arm. Override it to keep
# several runs of the SAME arm side by side under one SAMPLE_ROOT — most often one
# e2e-ttt-fomaml directory per phi_0 checkpoint. Without it a second checkpoint would
# write into the first one's directory and --skip_existing would skip all 100 clips,
# silently "finishing" instantly and leaving you comparing a checkpoint against itself.
# score-demos.sh takes the same variable, so pass it to both stages.
OUT_DIR="$SAMPLE_ROOT/${RUN_NAME:-$ARM}"
mkdir -p "$OUT_DIR"

# Materialise this shard's prompts. Round-robin (NR-1 mod N) rather than a contiguous
# block so every shard gets a similar mix and they finish at similar times.
if [ "$NUM_SHARDS" -gt 1 ]; then
    SHARD_FILE="$OUT_DIR/.shard-${SHARD_INDEX}-of-${NUM_SHARDS}.txt"
    awk -v i="$SHARD_INDEX" -v n="$NUM_SHARDS" \
        'NF && $0 !~ /^[[:space:]]*#/ { if (c++ % n == i) print }' \
        "$PROMPT_FILE" > "$SHARD_FILE"
else
    SHARD_FILE="$PROMPT_FILE"
fi
NUM_PROMPTS="$(grep -cve '^\s*$' -e '^\s*#' "$SHARD_FILE" || true)"

echo "Arm:        $ARM${RUN_NAME:+  (run: $RUN_NAME)}"
echo "Prompts:    $NUM_PROMPTS (shard $SHARD_INDEX/$NUM_SHARDS of $PROMPT_FILE)"
echo "Geometry:   ${HEIGHT}x${WIDTH}, $NUM_CHUNKS chunks"
echo "Output:     $OUT_DIR/<prompt[:30]>/"
[ "$ARM" = "e2e-ttt-fomaml" ] && echo "LoRA:       $LORA"
echo

COMMON=(
    --prompt_file "$SHARD_FILE"
    --negative_prompt "$NEGATIVE_PROMPT"
    --height "$HEIGHT" --width "$WIDTH"
    --seed "$SEED"
    --output-dir "$OUT_DIR"
    --skip_existing
)

case "$ARM" in
    base)
        python "$DRIVERS/Wan2.2-TI2V-5B-base-custom.py" \
            "${COMMON[@]}" --num_frames "$BASE_NUM_FRAMES"
        ;;
    chunk-by-chunk)
        python "$DRIVERS/Wan2.2-TI2V-5B-chunk-by-chunk-custom.py" \
            "${COMMON[@]}" --num_chunks "$NUM_CHUNKS" --frames_per_chunk 41
        ;;
    chunk-by-chunk-anchored)
        python "$DRIVERS/Wan2.2-TI2V-5B-chunk-by-chunk-anchored-custom.py" \
            "${COMMON[@]}" --num_chunks "$NUM_CHUNKS" --frames_per_chunk 53 \
            --num_anchor_latent_frames 3
        ;;
    e2e-ttt-fomaml)
        # These inner-loop flags must mirror the phi_0 checkpoint's training config
        # (Wan2.2-TI2V-5B-e2e-ttt-fomaml-k3-480p.yaml): adamw, 2 gradient steps,
        # rank 128, k=3, inner lr 1e-5. A mismatch silently measures a different
        # adaptation rule than the one that was meta-trained.
        python "$DRIVERS/Wan2.2-TI2V-5B-e2e-ttt-custom.py" \
            "${COMMON[@]}" --num_chunks "$NUM_CHUNKS" --frames_per_chunk 53 \
            --algorithm fomaml --lora "$LORA" \
            --optimizer adamw --num_gradient_steps 2 --lora_rank 128 \
            --num_anchor_latent_frames 3 --inner_lr_init 1e-5
        ;;
esac

echo
echo "Sampling done. Score this arm with:"
echo "  bash eval/visionreward/score-demos.sh $ARM"
