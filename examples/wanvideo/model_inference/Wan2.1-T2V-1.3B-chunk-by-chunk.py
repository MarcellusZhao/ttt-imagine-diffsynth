"""
Chunk-by-chunk long-video generation — Wan2.1-T2V-1.3B (base model, no adaptation).

This is the *baseline* counterpart to the E2E-TTT script
(`Wan2.1-T2V-1.3B-e2e-ttt.py`): for a single narrative prompt, the long video is
produced by generating `NUM_CHUNKS` contiguous sub-clips one after another and
concatenating them. Unlike E2E-TTT, the base model does NOT memorize the preceding
chunks and its weights are never modified — every chunk is an independent
text-to-video generation. It exists to show what long-video generation looks like
without any test-time training / LoRA "memory scratchpad".

`CONDITION_ON_LAST_CHUNK` optionally adds continuity between chunks: each chunk
after the first is seeded with the *entire previous chunk* and generated via the
video-to-video path (the previous chunk's frames are VAE-encoded and partially
renoised to `DENOISING_STRENGTH`, so generation continues from them). Note
Wan2.1-T2V-1.3B is a pure T2V model with no native image conditioning, so this
uses the V2V seed-video path rather than true I2V `input_image` conditioning
(which is a no-op on this DiT). With the flag off, chunks are fully independent.
"""

import torch
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

LOCAL_CKPT = "/work/nlp/hzhao/checkpoints/wan/Wan2.1-T2V-1.3B"

# Chunk-by-chunk controls (mirror InferenceConfig in the E2E-TTT script).
NUM_CHUNKS = 2
FRAMES_PER_CHUNK = 13   # 4n+1, aligns with the Wan temporal VAE compression
HEIGHT = 352
WIDTH = 512
NUM_INFERENCE_STEPS = 50
CFG_SCALE = 5.0
SIGMA_SHIFT = 5.0
SEED = 0
TILED = True

# Condition each chunk (after the first) on the entire previous chunk for
# continuity, via the video-to-video path. When False, every chunk is independent.
CONDITION_ON_LAST_CHUNK = True
# V2V renoising strength for conditioned chunks: 1.0 ignores the seed chunk
# entirely (pure T2V), lower values keep more of the previous chunk. ~0.7 trades
# continuity against motion freedom.
DENOISING_STRENGTH = 0.7

PROMPT = "纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。"
# PROMPT_LIST = [
#     "纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。",
#     "纪实摄影风格画面，一只活泼的小狗戴着黑色墨镜在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，戴着黑色墨镜，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。",
# ]
NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path=f"{LOCAL_CKPT}/diffusion_pytorch_model.safetensors"),
        ModelConfig(path=f"{LOCAL_CKPT}/models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(path=f"{LOCAL_CKPT}/Wan2.1_VAE.pth"),
    ],
    tokenizer_config=ModelConfig(path=f"{LOCAL_CKPT}/google/umt5-xxl"),
)

# Generate chunk by chunk. The base model never memorizes past chunks and its
# weights are never modified. The only optional carry-over is the previous chunk
# itself, fed back as a V2V seed when CONDITION_ON_LAST_CHUNK is on.
all_frames = []
prev_chunk = None
for k in range(NUM_CHUNKS):
    call_kwargs = dict(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        height=HEIGHT,
        width=WIDTH,
        num_frames=FRAMES_PER_CHUNK,
        num_inference_steps=NUM_INFERENCE_STEPS,
        cfg_scale=CFG_SCALE,
        sigma_shift=SIGMA_SHIFT,
        tiled=TILED,
        seed=SEED + k,
    )
    # Seed this chunk with the entire previous chunk and partially renoise it,
    # so generation continues from the previous chunk's content/motion.
    if CONDITION_ON_LAST_CHUNK and prev_chunk is not None:
        call_kwargs["input_video"] = prev_chunk
        call_kwargs["denoising_strength"] = DENOISING_STRENGTH

    frames = pipe(**call_kwargs)
    all_frames.extend(frames)
    prev_chunk = frames
    print(f"[chunk-by-chunk] generated chunk {k + 1}/{NUM_CHUNKS} ({len(frames)} frames)"
          + (" [conditioned on prev chunk]" if CONDITION_ON_LAST_CHUNK and k > 0 else ""))

save_video(all_frames, "video_chunk_by_chunk_Wan2.1-T2V-1.3B.mp4", fps=15, quality=5)
print(f"Saved a {len(all_frames)}-frame chunk-by-chunk long video.")
