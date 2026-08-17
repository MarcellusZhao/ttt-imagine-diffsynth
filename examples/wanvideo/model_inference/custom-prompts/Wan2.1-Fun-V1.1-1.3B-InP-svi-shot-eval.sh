#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SVI toy_test "shot" example — Wan2.1-Fun-V1.1-1.3B-InP, i2v anchor route.
#
#   https://github.com/vita-epfl/Stable-Video-Infinity/tree/main/data/toy_test/shot
#
# The point of this example is that it is IMAGE-CONDITIONED. Every arm here runs
# with --input_image, which is the regime SVI actually ships: `test_svi.py` reads a
# reference image off disk and NEVER has a chunk without conditioning frames
# (test_svi.py:374-382). Chunk 0's motion window is that image repeated m times
# (SVI's --repeat_first_clip), and the same image is the fixed reference filling
# `y`'s non-motion slots for the whole video (--ref_pad_num -1). Our other eval
# script samples without an image, so chunk 0 there falls into the null-conditioning
# regime, which meta-training only ever visits as a memorize input.
#
# Usage:  Wan2.1-Fun-V1.1-1.3B-InP-svi-shot-eval.sh <arm> [num_chunks]
#   arm ∈ base | chunk-by-chunk | chunk-by-chunk-anchored | e2e-ttt
#
# Unlike the sibling Wan2.1-Fun-V1.1-1.3B-InP-custom-prompt-eval.sh this takes the
# arm as an argument rather than by commenting lines, so two arms can be submitted
# concurrently without racing on the file's contents (the .sbatch reads it at job
# start, not at submit time).
#
# GEOMETRY is identical to that script's — fpc=45 / m=5 → stride 40 → 24 chunks =
# 965 frames — except arm 2, which trims 1 frame instead of 5 and so runs fpc=41 to
# land on the same stride. Do not change fpc / m / ref_pad_num / resolution on the
# TTT arm without changing them on the phi_0 checkpoint.
# ─────────────────────────────────────────────────────────────────────────────
set -e

ARM="${1:?usage: $0 <base|chunk-by-chunk|chunk-by-chunk-anchored|e2e-ttt> [num_chunks]}"
NUM_CHUNKS="${2:-24}"

SCRIPT_DIR="/home/hzhao/ttt-imagine-diffsynth/examples/wanvideo/model_inference/custom-prompts"
SVI_ROOT="${SVI_ROOT:-/home/hzhao/Stable-Video-Infinity}"

# The example itself. prompt.txt there is a Python literal (`prompts = [...]`), not a
# one-per-line file, so the string is inlined rather than read with --prompt_file.
IMAGE="$SVI_ROOT/data/toy_test/shot/frame.jpg"
PROMPT="A sleek white motor yacht speeds across the turquoise blue sea, leaving a dramatic wake of white foam behind it under a clear blue sky."
# SVI's own common_negative_prompt (test_svi.py:236) — the English twin of the
# Chinese one the other eval script uses.
NEGATIVE_PROMPT="bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

OUT_DIR="${OUT_DIR:-./results/svi-shot-1.3B/$NUM_CHUNKS-chunks}"

# phi_0 checkpoint (e2e-ttt arm only). Its inner-loop flags below MUST mirror it.
CKPT_DIR="${CKPT_DIR:-/work/nlp/hzhao/checkpoints/e2e-ttt/Wan2.1-Fun-V1.1-1.3B-InP_e2e_ttt_fomaml_i2v_m5_480p_fpc45_r128_reffirst_antidrift_svi_uvl_fs_rsfps16_len_grouped_3k-20260814-145829}"
LORA="${LORA:-$CKPT_DIR/step-375.safetensors}"

COMMON=(--prompt "$PROMPT" --negative_prompt "$NEGATIVE_PROMPT" --input_image "$IMAGE"
        --output-dir "$OUT_DIR" --height 480 --width 832)

case "$ARM" in
  # Single pass, no chunking. 45 + (N-1)*40 frames, i.e. the same length as the
  # chunked arms at the same chunk count.
  base)
    python "$SCRIPT_DIR/Wan2.1-Fun-V1.1-1.3B-InP-base-custom.py" "${COMMON[@]}" \
      --num_frames $((45 + (NUM_CHUNKS - 1) * 40))
    ;;
  # Stock single-frame I2V continuity — the motion-unidentifiable anchor. fpc=41,
  # not 45: this arm trims 1 frame, so 41 is what gives it stride 40.
  chunk-by-chunk)
    python "$SCRIPT_DIR/Wan2.1-Fun-V1.1-1.3B-InP-chunk-by-chunk-custom.py" "${COMMON[@]}" \
      --num_chunks "$NUM_CHUNKS" --frames_per_chunk 41
    ;;
  # Full E2E-TTT i2v anchoring on base weights, no LoRA — the no-adaptation ablation
  # of the arm below.
  chunk-by-chunk-anchored)
    python "$SCRIPT_DIR/Wan2.1-Fun-V1.1-1.3B-InP-chunk-by-chunk-anchored-custom.py" "${COMMON[@]}" \
      --num_chunks "$NUM_CHUNKS" --frames_per_chunk 45 --num_motion_frames 5 --ref_pad_num -1
    ;;
  # E2E-TTT. --optimizer adamw (the CLI default is sgd), --inner_lr_init 1e-5,
  # --num_gradient_steps 2, --lora_rank 128, m=5, ref_pad_num=-1 all mirror the checkpoint.
  e2e-ttt)
    python "$SCRIPT_DIR/Wan2.1-Fun-V1.1-1.3B-InP-e2e-ttt-custom.py" "${COMMON[@]}" \
      --num_chunks "$NUM_CHUNKS" --frames_per_chunk 45 --algorithm "fomaml" --lora "$LORA" \
      --output_name "Wan2.1-Fun-V1.1-1.3B-InP_e2e_ttt_fomaml_i2v_m5_480p_fpc45_r128_reffirst_antidrift_svi_3k.mp4" \
      --num_gradient_steps 2 --optimizer "adamw" --inner_lr_init 1e-5 --lora_rank 128 \
      --num_motion_frames 5 --ref_pad_num -1
    ;;
  *)
    echo "unknown arm '$ARM' (base | chunk-by-chunk | chunk-by-chunk-anchored | e2e-ttt)" >&2
    exit 1
    ;;
esac
