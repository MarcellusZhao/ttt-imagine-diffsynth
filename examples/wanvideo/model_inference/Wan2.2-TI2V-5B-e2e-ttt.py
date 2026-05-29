"""
End-to-End Test-Time Training (E2E-TTT) sequential long-video generation — Wan2.2-TI2V-5B.

For a single narrative prompt, the video is generated chunk by chunk: generate chunk k,
*memorize* it with in-place first-order LoRA updates, then generate chunk k+1 with the
adapted LoRA. The LoRA scratchpad is reset to the meta-init phi_0 before each narrative.

TI2V-5B can also condition the first chunk on an image: pass `input_image=...` to
`generator.generate(...)` (the remaining chunks continue purely from the adapted LoRA).

If you have a meta-trained LoRA (from `model_training/lora/Wan2.2-TI2V-5B-e2e-ttt.sh`),
point E2E_TTT_LORA at the saved checkpoint; otherwise generation starts from a zero-init
(identity) adapter and still adapts at test time.
"""

import os
import glob
import torch
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion.e2e_ttt import (
    InnerLoopConfig, InferenceConfig, inject_lora_for_ttt, WanE2ETTTSequentialGenerator,
)

LOCAL_CKPT = "/work/nlp/hzhao/checkpoints/wan/Wan2.2-TI2V-5B"
E2E_TTT_LORA = os.environ.get("E2E_TTT_LORA", None)  # optional meta-trained LoRA phi_0

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path=f"{LOCAL_CKPT}/models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(path=sorted(glob.glob(f"{LOCAL_CKPT}/diffusion_pytorch_model*.safetensors"))),
        ModelConfig(path=f"{LOCAL_CKPT}/Wan2.2_VAE.pth"),
    ],
    tokenizer_config=ModelConfig(path=f"{LOCAL_CKPT}/google/umt5-xxl"),
)

# Inject the LoRA "memory scratchpad" and (optionally) load the meta-trained phi_0.
phi0 = inject_lora_for_ttt(
    pipe,
    lora_rank=32,
    target_modules="q,k,v,o,ffn.0,ffn.2",
    lora_checkpoint=E2E_TTT_LORA if (E2E_TTT_LORA and os.path.exists(E2E_TTT_LORA)) else None,
)

inner_cfg = InnerLoopConfig(
    num_gradient_steps=1,
    num_mc_samples=2,
    inner_lr_init=5e-5,
    max_inner_grad_norm=1.0,
)
infer_cfg = InferenceConfig(
    num_chunks=4,
    frames_per_chunk=49,   # 4n+1
    ttt_steps_per_chunk=1,
    height=704,
    width=1248,
    num_inference_steps=50,
    cfg_scale=5.0,
    sigma_shift=5.0,
    seed=0,
    tiled=True,
)

generator = WanE2ETTTSequentialGenerator(pipe, inner_cfg, infer_cfg, phi0=phi0)

frames = generator.generate(
    prompt="两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    # input_image=Image.open("first_frame.jpg").resize((1248, 704)),  # optional I2V seed
)

save_video(frames, "video_e2e_ttt_Wan2.2-TI2V-5B.mp4", fps=15, quality=5)
print(f"Saved a {len(frames)}-frame E2E-TTT long video.")
