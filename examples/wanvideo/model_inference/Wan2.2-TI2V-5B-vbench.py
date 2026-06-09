"""
VBench prompt-suite sampling for Wan2.2-TI2V-5B (text-to-video mode).

Generates videos for the standard VBench text-to-video prompt suite, following
VBench's sampling protocol (see VBench/prompts/README.md):

  * Read prompts from `prompts/all_dimension.txt` (default) or from a single
    per-dimension file under `prompts/prompts_per_dimension/<dim>.txt`.
  * Sample N videos per prompt (default 5; VBench requires 25 for the
    `temporal_flickering` dimension to survive its static filter).
  * Use a *different, reproducible* seed for every sampled video.
  * Name each clip `<save_path>/<prompt>-<index>.mp4`, index in 0..N-1 — this
    exact filename is what the VBench evaluator parses to recover the prompt.

Wan2.2-TI2V-5B is a unified text+image-to-video model; for the VBench *T2V*
prompt suite we drive it in pure text-to-video mode (no input_image), matching
the text-to-video example in Wan2.2-TI2V-5B.py.

Because several VBench dimensions share the same prompt file, sampling once
from `all_dimension.txt` into a single folder covers 15 of the 16 dimensions;
only `temporal_flickering` needs its extra indices topped up afterwards (the
companion .sh launcher does this with --skip_existing).

Multi-GPU sharding: split the prompt list with --start_index/--end_index and
run one process per GPU, all writing to the same --save_path.

Evaluate afterwards, e.g.:
    vbench evaluate --videos_path <save_path> --dimension object_class
"""

import argparse
import glob
import json
import os

import torch
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
DEFAULT_MODEL_DIR = "/work/nlp/hzhao/checkpoints/wan/Wan2.2-TI2V-5B"
DEFAULT_VBENCH_ROOT = "/home/hzhao/VBench"

TEMPORAL_FLICKERING = "temporal_flickering"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample the VBench prompt suite with Wan2.2-TI2V-5B (T2V mode)."
    )

    # Config (mirrors the training entrypoint: a YAML whose leaf keys match these
    # argparse dest names; CLI flags still override the YAML).
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML inference config (see model_inference/configs/).")

    # Model loading
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR,
                        help="Directory containing the Wan2.2-TI2V-5B checkpoint files. "
                             "Used only when --model_paths is not given.")
    parser.add_argument("--model_paths", type=str, default=None,
                        help="JSON list of weight paths (overrides --model_dir). A nested "
                             "list element loads as one sharded ModelConfig (the 5B DiT).")
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
                        help="Videos to sample per prompt. Defaults to 5, or 25 when "
                             "--dimension temporal_flickering.")
    parser.add_argument("--base_seed", type=int, default=0,
                        help="Seed for video index i is base_seed + i (reproducible, distinct).")
    parser.add_argument("--start_index", type=int, default=0,
                        help="First prompt (inclusive) to sample — for multi-GPU sharding.")
    parser.add_argument("--end_index", type=int, default=None,
                        help="Last prompt (exclusive) to sample — for multi-GPU sharding.")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip a clip if its output file already exists (resumable).")

    # Generation settings (Wan2.2-TI2V-5B native 720p)
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT,
                        help="Negative prompt (applied to every clip).")
    parser.add_argument("--height", type=int, default=704, help="Output video height.")
    parser.add_argument("--width", type=int, default=1248, help="Output video width.")
    parser.add_argument("--num_frames", type=int, default=121,
                        help="Frames per clip (4n+1 for the Wan temporal VAE).")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of denoising steps.")
    parser.add_argument("--cfg_scale", type=float, default=5.0,
                        help="Classifier-free guidance scale.")

    # Output
    parser.add_argument("--save_path", type=str, required=True,
                        help="Directory for sampled clips, named <prompt>-<index>.mp4.")
    parser.add_argument("--fps", type=int, default=24, help="Output video FPS.")
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
    arbitrary sections (e.g. `model:`, `vbench:`, `generation:`, `output:`) for
    readability; only leaf keys matter, and they must match argparse dest names
    (e.g. `num_videos_per_prompt`, `cfg_scale`). CLI flags still override the YAML.
    Returns the set of unrecognised leaf keys (for a warning). Mirrors
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
    """text-encoder / DiT / VAE ModelConfigs from --model_paths (JSON list; a
    nested element loads as one sharded ModelConfig — the 5B DiT is 3 shards) or,
    failing that, from the canonical filenames under --model_dir."""
    if args.model_paths:
        paths = json.loads(args.model_paths) if isinstance(args.model_paths, str) else args.model_paths
        return [ModelConfig(path=p) for p in paths]
    return [
        ModelConfig(path=os.path.join(args.model_dir, "models_t5_umt5-xxl-enc-bf16.pth")),
        ModelConfig(path=sorted(glob.glob(os.path.join(
            args.model_dir, "diffusion_pytorch_model*.safetensors")))),
        ModelConfig(path=os.path.join(args.model_dir, "Wan2.2_VAE.pth")),
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
    print(f"[VBench/Wan2.2-TI2V-5B] {len(shard)} prompts "
          f"(indices {args.start_index}:{end_index} of {len(prompts)}) "
          f"x {num_videos} videos -> {args.save_path}")

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

            video = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                seed=args.base_seed + index,
                tiled=True,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.cfg_scale,
            )
            save_video(video, out_path, fps=args.fps, quality=args.quality)
            print(f"  [{args.start_index + p_i + 1}/{end_index}] "
                  f"seed={args.base_seed + index} -> {out_path}")


if __name__ == "__main__":
    main()
