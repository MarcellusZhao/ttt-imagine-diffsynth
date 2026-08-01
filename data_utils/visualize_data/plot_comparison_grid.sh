RESULT_DIR="/home/hzhao/ttt-imagine-diffsynth/results/custom-prompts/20-chunks/两只小松鼠穿着运动背心，在公园里进行跑步比赛。它们沿着弯曲的"

python -m plot_comparison_grid \
    --videos "$RESULT_DIR/Wan2.2-TI2V-5B.mp4" \
             "$RESULT_DIR/Wan2.2-TI2V-5B-chunk-by-chunk-with-conditioning.mp4" \
             "$RESULT_DIR/Wan2.2-TI2V-5B_e2e_ttt_fomaml_cosine_lr_adamw_gs2_latent_handoff_err_rec_acn_uvl_fs_rsfps16_11k_4gpu.mp4" \
    --labels One-Pass-Gen Chunk-by-Chunk Ours \
    --start 0 --step 110 --count 7 \
    --fps 16 \
    --output "$RESULT_DIR/comparison.pdf"


# Memory Test
# RESULT_DIR="/home/hzhao/ttt-imagine-diffsynth/results/memory-test/bear_leave_return_v2"

# python -m plot_comparison_grid \
#     --videos "$RESULT_DIR/base-model.mp4" \
#              "$RESULT_DIR/chunk-by-chunk.mp4" \
#              "$RESULT_DIR/e2e-ttt-fomaml-with-conditioning.mp4" \
#     --labels One-Pass-Gen Chunk-by-Chunk Ours \
#     --start 0 --step 40 --count 7 \
#     --fps 16 \
#     --output "$RESULT_DIR/comparison.pdf"
