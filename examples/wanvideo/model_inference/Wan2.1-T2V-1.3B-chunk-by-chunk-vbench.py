"""
VBench prompt-suite sampling with CHUNK-BY-CHUNK long-video generation — Wan2.1-T2V-1.3B.

This is the chunk-by-chunk counterpart to Wan2.1-T2V-1.3B-vbench.py: instead of one
single-shot text-to-video call per clip, each VBench clip is produced by generating
`--num_chunks` contiguous sub-clips and concatenating them (the same procedure as
Wan2.1-T2V-1.3B-chunk-by-chunk-custom.py, just driven over the whole VBench suite).
The base model never memorizes past chunks; the only carry-over is the optional
V2V seed described below.

VBench sampling protocol (see VBench/prompts/README.md):
  * Read prompts from `prompts/all_dimension.txt` (default) or a single per-dimension
    file under `prompts/prompts_per_dimension/<dim>.txt`.
  * Sample N clips per prompt (default 5; 25 for `temporal_flickering`).
  * Use a *different, reproducible* seed for every clip — and, within a clip, a
    distinct seed per chunk: clip `index` uses chunk seeds
    `base_seed + index*num_chunks + k`, so no two chunks across the whole run collide.
  * Name each clip `<save_path>/<prompt>-<index>.mp4`, index in 0..N-1 — the exact
    filename the VBench evaluator parses to recover the prompt.

Inter-chunk conditioning (`--condition_on_last_chunk`, on by default): each chunk
after the first is seeded with the *entire previous chunk* via the V2V path (the
previous chunk's frames are VAE-encoded and partially renoised to
`--denoising_strength`), so generation continues from them. Wan2.1-T2V-1.3B is a
pure T2V model with no native image conditioning, so this uses the V2V seed-video
path rather than true I2V. With the flag off, every chunk is independent T2V.

A YAML `--config` (see model_inference/configs/) can supply any of these settings;
CLI flags still override it. Multi-GPU sharding: split with --start_index/--end_index.

Evaluate afterwards, e.g.:
    vbench evaluate --videos_path <save_path> --dimension object_class
"""

import argparse
import json
import os

import torch
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
DEFAULT_MODEL_DIR = "/work/nlp/hzhao/checkpoints/wan/Wan2.1-T2V-1.3B"
DEFAULT_VBENCH_ROOT = "/home/hzhao/VBench"

TEMPORAL_FLICKERING = "temporal_flickering"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chunk-by-chunk VBench prompt-suite sampling with Wan2.1-T2V-1.3B."
    )

    # Config (mirrors the training entrypoint: a YAML whose leaf keys match these
    # argparse dest names; CLI flags still override the YAML).
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML inference config (see model_inference/configs/).")

    # Model loading
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR,
                        help="Directory containing the Wan2.1-T2V-1.3B checkpoint files. "
                             "Used only when --model_paths is not given.")
    parser.add_argument("--model_paths", type=str, default=None,
                        help="JSON list of weight paths (overrides --model_dir). A nested "
                             "list element loads as one sharded ModelConfig.")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to the umt5-xxl tokenizer (defaults to <model_dir>/google/umt5-xxl).")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to load the pipeline on.")

    # VBench prompt selection
    parser.add_argument("--vbench_root", type=str, default=DEFAULT_VBENCH_ROOT,
                        help="Path to the VBench repository.")
    parser.add_argument("--dimension", type=str, default="all",
                        help="'all' to read prompts/all_dimension.txt, or a dimension "
                             "name to read prompts/prompts_per_dimension/<dimension>.txt "
                             "(e.g. object_class, temporal_flickering).")
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="Explicit path to a prompt .txt (one prompt per line). "
                             "Overrides --dimension when set.")

    # Sampling protocol
    parser.add_argument("--num_videos_per_prompt", type=int, default=None,
                        help="Clips to sample per prompt. Defaults to 5, or 25 when "
                             "--dimension temporal_flickering.")
    parser.add_argument("--base_seed", type=int, default=0,
                        help="Clip `index` uses chunk seeds base_seed + index*num_chunks + k "
                             "(reproducible, collision-free across the run).")
    parser.add_argument("--start_index", type=int, default=0,
                        help="First prompt (inclusive) to sample — for multi-GPU sharding.")
    parser.add_argument("--end_index", type=int, default=None,
                        help="Last prompt (exclusive) to sample — for multi-GPU sharding.")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip a clip if its output file already exists (resumable).")

    # Chunk-by-chunk controls
    parser.add_argument("--num_chunks", type=int, default=2,
                        help="Number of contiguous sub-clips to generate and concatenate.")
    parser.add_argument("--frames_per_chunk", type=int, default=13,
                        help="Frames per chunk (4n+1, aligns with the Wan temporal VAE compression).")
    parser.add_argument("--no_condition_on_last_chunk", dest="condition_on_last_chunk",
                        action="store_false",
                        help="Disable seeding each chunk on the previous chunk (V2V continuity). "
                             "Enabled by default; when disabled, every chunk is independent.")
    parser.add_argument("--denoising_strength", type=float, default=0.7,
                        help="V2V renoising strength for conditioned chunks: 1.0 ignores the seed "
                             "chunk entirely (pure T2V), lower values keep more of the previous chunk.")

    # Generation settings
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT,
                        help="Negative prompt (applied to every chunk).")
    parser.add_argument("--height", type=int, default=480, help="Output video height.")
    parser.add_argument("--width", type=int, default=832, help="Output video width.")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of denoising steps.")
    parser.add_argument("--cfg_scale", type=float, default=5.0,
                        help="Classifier-free guidance scale.")
    parser.add_argument("--sigma_shift", type=float, default=5.0,
                        help="Flow-matching sigma shift.")

    # Output
    parser.add_argument("--save_path", type=str, required=True,
                        help="Directory for sampled clips, named <prompt>-<index>.mp4.")
    parser.add_argument("--fps", type=int, default=16, help="Output video FPS.")
    parser.add_argument("--quality", type=int, default=5, help="Output video quality.")

    # Pre-parse only --config so the YAML can populate defaults before the real parse.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()
    if pre_args.config is not None:
        unknown = apply_yaml_config(parser, pre_args.config)
        if unknown:
            print(f"[VBench] WARNING: ignoring unknown config keys: {sorted(unknown)}")
    return parser.parse_args()


def apply_yaml_config(parser, config_path):
    """Load a YAML config and use its values as argument defaults so a single
    `--config foo.yaml` replaces the long CLI. The YAML may be grouped into
    arbitrary sections (e.g. `model:`, `vbench:`, `chunking:`, `generation:`,
    `output:`) for readability; only leaf keys matter, and they must match argparse
    dest names (e.g. `num_chunks`, `denoising_strength`). CLI flags still override
    the YAML. Returns the set of unrecognised leaf keys (for a warning). Mirrors
    train_e2e_ttt.apply_yaml_config."""
    import yaml

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    flat = {}
    def _walk(d):
        for k, v in d.items():
            if isinstance(v, dict):
                _walk(v)
            else:
                flat[k] = v
    _walk(cfg)

    # A model_paths list (possibly nested for sharded weights) is carried as the
    # JSON form --model_paths expects.
    if isinstance(flat.get("model_paths"), list):
        flat["model_paths"] = json.dumps(flat["model_paths"])

    valid_dests = {a.dest for a in parser._actions}
    defaults = {k: v for k, v in flat.items() if k in valid_dests}
    parser.set_defaults(**defaults)
    # A value supplied by the YAML satisfies a `required=True` argument, but argparse
    # enforces `required` regardless of defaults, so clear it for those args.
    for action in parser._actions:
        if action.dest in defaults and getattr(action, "required", False):
            action.required = False
    return set(flat) - valid_dests - {"config"}


def build_model_configs(args):
    """DiT / text-encoder / VAE ModelConfigs from --model_paths (JSON list; a
    nested element loads as one sharded ModelConfig) or, failing that, from the
    canonical filenames under --model_dir."""
    if args.model_paths:
        paths = json.loads(args.model_paths) if isinstance(args.model_paths, str) else args.model_paths
        return [ModelConfig(path=p) for p in paths]
    return [
        ModelConfig(path=os.path.join(args.model_dir, "diffusion_pytorch_model.safetensors")),
        ModelConfig(path=os.path.join(args.model_dir, "models_t5_umt5-xxl-enc-bf16.pth")),
        ModelConfig(path=os.path.join(args.model_dir, "Wan2.1_VAE.pth")),
    ]


def build_tokenizer_config(args):
    path = args.tokenizer_path or os.path.join(args.model_dir, "google/umt5-xxl")
    return ModelConfig(path=path)


def resolve_prompt_file(args):
    if args.prompt_file is not None:
        return args.prompt_file
    if args.dimension == "all":
        return os.path.join(args.vbench_root, "prompts", "all_dimension.txt")
    return os.path.join(args.vbench_root, "prompts", "prompts_per_dimension",
                        f"{args.dimension}.txt")


def read_prompts(prompt_file):
    with open(prompt_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def generate_chunked_video(pipe, args, prompt, seed_base):
    """Generate one VBench clip as num_chunks concatenated sub-clips. Chunk k uses
    seed_base + k; when conditioning is on, chunk k>0 is seeded with the whole
    previous chunk via the V2V path (input_video + denoising_strength)."""
    all_frames = []
    prev_chunk = None
    for k in range(args.num_chunks):
        call_kwargs = dict(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.frames_per_chunk,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            sigma_shift=args.sigma_shift,
            tiled=True,
            seed=seed_base + k,
        )
        if args.condition_on_last_chunk and prev_chunk is not None:
            call_kwargs["input_video"] = prev_chunk
            call_kwargs["denoising_strength"] = args.denoising_strength
        frames = pipe(**call_kwargs)
        all_frames.extend(frames)
        prev_chunk = frames
    return all_frames


def main():
    args = parse_args()

    num_videos = args.num_videos_per_prompt
    if num_videos is None:
        num_videos = 25 if args.dimension == TEMPORAL_FLICKERING else 5

    prompt_file = resolve_prompt_file(args)
    prompts = read_prompts(prompt_file)
    end_index = args.end_index if args.end_index is not None else len(prompts)
    shard = prompts[args.start_index:end_index]

    os.makedirs(args.save_path, exist_ok=True)
    print(f"[VBench/Wan2.1-T2V-1.3B chunk-by-chunk] {len(shard)} prompts "
          f"(indices {args.start_index}:{end_index} of {len(prompts)}) "
          f"x {num_videos} clips x {args.num_chunks} chunks -> {args.save_path}")

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=build_model_configs(args),
        tokenizer_config=build_tokenizer_config(args),
    )

    for p_i, prompt in enumerate(shard):
        for index in range(num_videos):
            out_path = os.path.join(args.save_path, f"{prompt}-{index}.mp4")
            if args.skip_existing and os.path.exists(out_path):
                continue

            seed_base = args.base_seed + index * args.num_chunks
            frames = generate_chunked_video(pipe, args, prompt, seed_base)
            save_video(frames, out_path, fps=args.fps, quality=args.quality)
            print(f"  [{args.start_index + p_i + 1}/{end_index}] clip {index} "
                  f"({len(frames)} frames, seeds {seed_base}..{seed_base + args.num_chunks - 1}) "
                  f"-> {out_path}")


if __name__ == "__main__":
    main()
