#!/usr/bin/env bash
# Memory-test evaluation driver — Wan2.2-TI2V-5B.
#
# Each CASE is a per-chunk prompt schedule (one sub-prompt per chunk) that scripts a
# character leaving the frame for a few chunks and then returning. The point is to probe
# the LoRA "memory" module: while the character is out of frame the last-frame pixel
# anchor cannot carry its appearance, so only the LoRA scratchpad can.
#
# For every case we generate four videos so the memory contribution is attributable:
#   1. e2e-ttt, last-frame conditioning ON   -> full deployed system (LoRA memory + anchor)
#   2. e2e-ttt, last-frame conditioning OFF  -> isolate the LoRA memory (no pixel anchor)
#   3. chunk-by-chunk baseline, conditioning ON -> control: no LoRA memory at all
#   4. base model, all sub-prompts combined into one long prompt -> control: no chunking
#
# Sub-prompt schedule convention (6 chunks): present (0-1) -> absent (2-3) -> return (4-5).
# Add a case by defining a new *_PROMPTS array and calling `run_case`.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

E2E_TTT_PY="$HERE/Wan2.2-TI2V-5B-e2e-ttt-memory.py"
CHUNK_PY="$HERE/Wan2.2-TI2V-5B-chunk-by-chunk-memory.py"
BASE_PY="$HERE/Wan2.2-TI2V-5B-base-memory.py"

# --- Meta-trained LoRA phi_0 checkpoint (point this at your own). ---
ALGORITHM="fomaml"
LORA="/home/hzhao/ttt-imagine-diffsynth/models/train/Wan2.2-TI2V-5B_e2e_ttt_fomaml_cosine_lr_adamw_gs2_latent_handoff_err_rec_acn_uvl_fs_rsfps16_11k_4gpu-20260722-195501/epoch-0.safetensors"
# Inner-loop knobs matching the checkpoint above (adamw, 2 gradient steps).
INNER_FLAGS=(--optimizer adamw --num_gradient_steps 2 --lora_rank 128)

FRAMES_PER_CHUNK=81 # 41, 81
# Base-model single-pass length (4n+1). Kept modest so the base model does not OOM;
# bump it toward num_chunks*FRAMES_PER_CHUNK for a length-matched comparison.
BASE_NUM_FRAMES=485 # 245, 485
OUTPUT_DIR="./results/memory-test"
NEGATIVE_PROMPT="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

# run_case <case_name> <sub-prompt> [<sub-prompt> ...]
run_case () {
    local case_name="$1"; shift
    local prompts=("$@")
    echo "=== memory-test case: ${case_name} (${#prompts[@]} chunks) ==="

    # 1. e2e-ttt WITH last-frame conditioning (full deployed system).
    python "$E2E_TTT_PY" \
        --case_name "$case_name" \
        --prompts "${prompts[@]}" \
        --negative_prompt "$NEGATIVE_PROMPT" \
        --algorithm "$ALGORITHM" --lora "$LORA" "${INNER_FLAGS[@]}" \
        --frames_per_chunk "$FRAMES_PER_CHUNK" \
        --output-dir "$OUTPUT_DIR" \
        --output_name "e2e-ttt-${ALGORITHM}-with-conditioning.mp4"

    # 2. e2e-ttt WITHOUT last-frame conditioning (isolate the LoRA memory).
    python "$E2E_TTT_PY" \
        --case_name "$case_name" \
        --prompts "${prompts[@]}" \
        --negative_prompt "$NEGATIVE_PROMPT" \
        --algorithm "$ALGORITHM" --lora "$LORA" "${INNER_FLAGS[@]}" \
        --frames_per_chunk "$FRAMES_PER_CHUNK" \
        --no_condition_on_last_frame --no_drop_boundary_frame \
        --output-dir "$OUTPUT_DIR" \
        --output_name "e2e-ttt-${ALGORITHM}-without-conditioning.mp4"

    # 3. chunk-by-chunk baseline WITH conditioning (control: no LoRA memory).
    python "$CHUNK_PY" \
        --case_name "$case_name" \
        --prompts "${prompts[@]}" \
        --negative_prompt "$NEGATIVE_PROMPT" \
        --frames_per_chunk "$FRAMES_PER_CHUNK" \
        --output-dir "$OUTPUT_DIR" \
        --output_name "chunk-by-chunk.mp4"

    # 4. base-model baseline: all sub-prompts combined into one long prompt, single pass
    #    (control: no chunking, no LoRA memory).
    python "$BASE_PY" \
        --case_name "$case_name" \
        --prompts "${prompts[@]}" \
        --negative_prompt "$NEGATIVE_PROMPT" \
        --num_frames "$BASE_NUM_FRAMES" \
        --output-dir "$OUTPUT_DIR" \
        --output_name "base-model.mp4"
}

# --- Case 1: an orange cat leaves the courtyard and returns; a corgi stays. ---
# The cat's markings (right-ear white patch, blue collar bell) are the appearance probe.
# CAT_PROMPTS=(
#     "阳光明媚的庭院里，一只毛色橘白相间、右耳有一小块白斑、脖子上系着蓝色小铃铛的蓬松橘猫，和一只棕白色的柯基犬在草地上一起玩耍。"
#     "那只橘白相间、右耳有白斑、戴蓝色铃铛的橘猫，继续和棕白色柯基犬在庭院草地上依偎，气氛轻松愉快。"
#     "橘猫转身沿着石板小路走出画面，离开了画面，只剩下那只棕白色的柯基犬独自趴在草地上。"
#     "空荡荡的庭院草地上只有那只棕白色柯基犬，它四处张望、来回走动，耐心地等待着同伴。"
#     "那只橘白相间、右耳有一小块白斑、脖子上系着蓝色小铃铛的橘猫，从石板小路尽头慢慢走回庭院，回到柯基犬身边，毛色和花纹和先前一模一样。"
#     "橘猫和棕白色柯基犬重新在庭院草地上开心地依偎、玩耍，画面温馨连贯。"
# )
CAT_PROMPTS=(
    "阳光明媚的庭院里，一只毛色橘白相间、右耳有一小块白斑、脖子上系着蓝色小铃铛的蓬松橘猫，和一只棕白色的柯基犬在草地上一起玩耍。"
    "那只橘猫继续和柯基犬在庭院草地上依偎，气氛轻松愉快。"
    "橘猫转身沿着石板小路走出画面，离开了画面，只剩下那只柯基犬独自趴在草地上。"
    "空荡荡的庭院草地上只有那只柯基犬，它四处张望、来回走动，耐心地等待着同伴。"
    "那只橘猫从石板小路尽头慢慢走回庭院，回到柯基犬身边。"
    "橘猫和柯基犬重新在庭院草地上开心地依偎、玩耍，画面温馨连贯。"
)

# --- Case 2: a little bear (blue knitted scarf) steps behind a tree and comes back. ---
# BEAR_PROMPTS=(
#     "森林空地上野餐，一只戴着蓝色针织围巾、胸前有一撮白毛的小棕熊，和一只小白兔坐在格子野餐垫上分享食物。"
#     "那只戴蓝色针织围巾、胸前有白毛的小棕熊，和小白兔在野餐垫上愉快地吃着点心，阳光透过树叶洒下。"
#     "小棕熊起身走到一棵大树后面，消失在画面之外，野餐垫上只剩下那只小白兔。"
#     "小白兔独自坐在野餐垫上，安静地张望四周，等待同伴回来。"
#     "那只戴着蓝色针织围巾、胸前有一撮白毛的小棕熊，从大树后面走回来，回到野餐垫旁，样子和先前完全一样。"
#     "小棕熊和小白兔重新一起在野餐垫上分享食物，气氛温馨。"
# )
BEAR_PROMPTS=(
    "森林空地上野餐，一只戴着蓝色针织围巾、胸前有一撮白毛的小棕熊，和一只小白兔坐在格子野餐垫上分享食物。"
    "那只小棕熊和小白兔在野餐垫上愉快地吃着点心，阳光透过树叶洒下。"
    "小棕熊起身走到一棵大树后面，消失在画面之外，野餐垫上只剩下那只小白兔。"
    "小白兔独自坐在野餐垫上，安静地张望四周，等待同伴回来。"
    "那只小棕熊从大树后面走回来，回到野餐垫旁，样子和先前完全一样。"
    "小棕熊和小白兔重新一起在野餐垫上分享食物，气氛温馨。"
)

run_case "cat_leave_return_v3" "${CAT_PROMPTS[@]}"
run_case "bear_leave_return_v3" "${BEAR_PROMPTS[@]}"
