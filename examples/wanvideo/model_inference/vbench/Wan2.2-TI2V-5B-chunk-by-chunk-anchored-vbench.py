"""
VBench prompt-suite sampling with CHUNK-BY-CHUNK generation and E2E-TTT-style anchoring
— Wan2.2-TI2V-5B (base model, no LoRA, no test-time adaptation).

This is the VBench driver for the ablation partner of the E2E-TTT runs: it is
`Wan2.2-TI2V-5B-chunk-by-chunk-anchored-custom.py` swept over the whole VBench suite,
i.e. `Wan2.2-TI2V-5B-e2e-ttt-vbench.py` with the LoRA memory scratchpad removed and
*everything else* — the chunk loop, the inter-chunk conditioning, the boundary trim —
held fixed. Read an E2E-TTT VBench number against this arm and the only difference is
the scratchpad.

Two anchors are pinned into each follow-up chunk's leading latents via TI2V-5B's fused
VAE conditioning:
  * a **wide local anchor** — the previous chunk's trailing window, handed forward as ONE
    contiguous clip (`anchor_frames`) so it encodes to a k-latent *block* that carries
    real velocity rather than a motion-ambiguous single frame.
    `--num_anchor_latent_frames` (k, default 3);
  * a **global anchor** — the very first frame of the whole clip (the "sink"), captured
    once after chunk 0 and reused unchanged, giving every chunk a non-sliding reference
    to how the clip started to counteract long-horizon drift.

With the sink on, the anchor block is displaced to latent positions 1..k, so the window
handed forward is 4k+1 pixel frames (k=3 -> 13) — one extra leading frame purely as VAE
causal context, whose latent the sink overwrites. Those pinned frames are context, not
new content, so they are trimmed at each boundary (`--no_drop_boundary_frame` keeps them).
`--no_condition_on_sink_frame` drops the global anchor; `--no_condition_on_last_frame`
drops both and makes every chunk an independent text-to-video generation.

Unlike the plain `-chunk-by-chunk-vbench.py` (single last frame via `input_image`, k=1
implicitly), the defaults here match the E2E-TTT scripts' anchoring, with the wider
`--num_anchor_latent_frames 3` the anchored-custom script also defaults to. Keep the
chunk geometry (`num_chunks`, `frames_per_chunk`, k, resolution) identical to the
E2E-TTT arm you are comparing against — VBench metrics are length-sensitive.

VBench sampling protocol (see VBench/prompts/README.md):
  * Read prompts from `prompts/all_dimension.txt` (default) or a single per-dimension
    file under `prompts/prompts_per_dimension/<dim>.txt`.
  * Sample N clips per prompt (default 5; 25 for `temporal_flickering`).
  * Use a *different, reproducible* seed for every clip — and, within a clip, a distinct
    seed per chunk: clip `index` uses chunk seeds `base_seed + index*num_chunks + k`, so
    no two chunks across the whole run collide (the same scheme as the other two drivers,
    so arms share seeds clip for clip).
  * Name each clip `<save_path>/<prompt>-<index>.mp4`, index in 0..N-1 — the exact
    filename the VBench evaluator parses to recover the prompt.

A YAML `--config` (see model_inference/configs/) can supply any of these settings;
CLI flags still override it. Multi-GPU sharding: split with --start_index/--end_index.

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
from diffsynth.diffusion.e2e_ttt import (
    anchor_overlap_pixel_frames, num_clean_latents, num_pinned_pixel_frames,
)


DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
DEFAULT_MODEL_DIR = "/work/nlp/hzhao/checkpoints/wan/Wan2.2-TI2V-5B"
DEFAULT_VBENCH_ROOT = "/home/hzhao/VBench"

TEMPORAL_FLICKERING = "temporal_flickering"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chunk-by-chunk VBench prompt-suite sampling with Wan2.2-TI2V-5B "
                    "(base model) using the E2E-TTT anchoring strategy "
                    "(wide local anchor + first-frame sink)."
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
    parser.add_argument("--num_chunks", type=int, default=4,
                        help="Number of contiguous sub-clips to generate and concatenate.")
    parser.add_argument("--frames_per_chunk", type=int, default=49,
                        help="Frames per chunk (4n+1, aligns with the Wan temporal VAE compression).")

    # Inter-chunk anchoring (mirrors Wan2.2-TI2V-5B-e2e-ttt-vbench.py)
    parser.add_argument("--no_condition_on_last_frame", dest="condition_on_last_frame",
                        action="store_false",
                        help="Disable anchoring each chunk on the previous chunk's trailing "
                             "window (enabled by default). Turning this off also drops the "
                             "sink, making every chunk an independent text-to-video generation.")
    parser.add_argument("--no_condition_on_sink_frame", dest="condition_on_sink_frame",
                        action="store_false",
                        help="Disable the fixed global anchor -- the clip's very first "
                             "generated frame -- pinned alongside the sliding local anchor "
                             "(enabled by default). Requires last-frame conditioning.")
    parser.add_argument("--no_drop_boundary_frame", dest="drop_boundary_frame",
                        action="store_false",
                        help="Keep the pinned anchor frames at each chunk boundary "
                             "(dropped by default).")
    parser.add_argument("--num_anchor_latent_frames", type=int, default=3,
                        help="Width k of the local anchor block in LATENT frames (default 3). "
                             "k>1 hands the previous chunk's trailing window forward as one "
                             "contiguous encode, so the model sees actual velocity instead of "
                             "a motion-ambiguous single frame. k=1 is the legacy single-frame "
                             "anchor. Match the E2E-TTT arm being compared against.")

    # Generation settings
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT,
                        help="Negative prompt (applied to every chunk).")
    parser.add_argument("--height", type=int, default=704, help="Output video height.")
    parser.add_argument("--width", type=int, default=1280, help="Output video width.")
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
    dest names (e.g. `num_chunks`, `num_anchor_latent_frames`). CLI flags still
    override the YAML. Returns the set of unrecognised leaf keys (for a warning).
    Mirrors train_e2e_ttt.apply_yaml_config."""
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


def generate_anchored_video(pipe, args, prompt, seed_base, k_anchor,
                            condition_on_sink_frame, anchor_window):
    """Generate one VBench clip as num_chunks concatenated sub-clips, anchored exactly
    the way the E2E-TTT generator anchors — a contiguous k-latent block from the
    previous chunk's tail plus the clip's first frame as a fixed sink. Chunk k uses
    seed_base + k. Pinned (context, not content) frames are trimmed at each boundary."""
    all_frames = []
    # Running local anchor: the previous chunk's trailing `anchor_window` frames, re-encoded
    # as ONE contiguous clip by WanVideoUnit_ImageEmbedderFused so the anchor latents land at
    # the positions they belong to. k=1 keeps the legacy single-frame `input_image` path.
    cond_image = None
    cond_frames = None
    # Fixed global anchor: the clip's first raw frame, captured once after chunk 0.
    sink_image = None
    for k in range(args.num_chunks):
        call_kwargs = dict(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            seed=seed_base + k,
            tiled=True,
            height=args.height,
            width=args.width,
            num_frames=args.frames_per_chunk,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            sigma_shift=args.sigma_shift,
        )
        # Anchor this follow-up chunk on the previous chunk's tail.
        if k > 0 and args.condition_on_last_frame:
            if k_anchor > 1 and cond_frames is not None:
                # `anchor_frames` supersedes `input_image` inside the fused embedder.
                call_kwargs["anchor_frames"] = cond_frames
            elif cond_image is not None:
                call_kwargs["input_image"] = cond_image
        sink_active = (
            k > 0 and condition_on_sink_frame
            and args.condition_on_last_frame and sink_image is not None
        )
        if sink_active:
            call_kwargs["sink_image"] = sink_image

        frames = pipe(**call_kwargs)

        # The leading frames of a conditioned follow-up chunk reproduce the pinned anchor(s)
        # rather than new content; drop them to avoid a duplicate-frame seam at the boundary.
        if k > 0 and args.condition_on_last_frame and args.drop_boundary_frame:
            emitted = frames[num_pinned_pixel_frames(num_clean_latents(k_anchor, sink_active)):]
        else:
            emitted = frames
        all_frames.extend(emitted)

        # Carry the local anchor forward: the trailing window for the k>1 block path, the
        # single last frame for the legacy k=1 path.
        if args.condition_on_last_frame and len(frames) > 0:
            cond_image = frames[-1]
            cond_frames = frames[-anchor_window:] if k_anchor > 1 else None
            if cond_frames is not None and len(cond_frames) < anchor_window:
                # Would silently encode to fewer than k anchor latents. Only reachable with a
                # chunk shorter than the anchor window, i.e. a misconfigured frames_per_chunk.
                raise ValueError(
                    f"chunk {k} produced {len(frames)} frames but the k={k_anchor} anchor "
                    f"block needs {anchor_window}; raise --frames_per_chunk or lower "
                    f"--num_anchor_latent_frames."
                )
        # Capture the clip's first raw frame once, as the fixed sink anchor.
        if k == 0 and sink_image is None and len(frames) > 0:
            sink_image = frames[0]
    return all_frames


def main():
    args = parse_args()

    # The sink is a *second* fused clean frame pinned next to the sliding local anchor, so it
    # only exists on the last-frame conditioning path (same normalization as the E2E-TTT
    # scripts and train_e2e_ttt.py).
    condition_on_sink_frame = args.condition_on_sink_frame and args.condition_on_last_frame
    if args.condition_on_sink_frame and not args.condition_on_last_frame:
        print("[VBench] NOTE: the first-frame sink requires last-frame conditioning; "
              "sampling without the sink.")

    k_anchor = max(1, int(args.num_anchor_latent_frames))
    # Pixel frames of each chunk handed forward as the next chunk's anchor block, and the
    # number of leading decoded frames that are pinned context rather than new content.
    anchor_window = anchor_overlap_pixel_frames(k_anchor, condition_on_sink_frame)
    n_clean = num_clean_latents(k_anchor, condition_on_sink_frame)
    pinned = num_pinned_pixel_frames(n_clean)
    if args.condition_on_last_frame and args.frames_per_chunk <= max(anchor_window, pinned):
        raise ValueError(
            f"frames_per_chunk={args.frames_per_chunk} is too short for a k={k_anchor} anchor "
            f"block (needs {anchor_window} frames handed forward, {pinned} pinned per chunk); "
            f"raise --frames_per_chunk or lower --num_anchor_latent_frames."
        )

    num_videos = args.num_videos_per_prompt
    if num_videos is None:
        num_videos = 25 if args.dimension == TEMPORAL_FLICKERING else 5

    prompt_file = resolve_prompt_file(args)
    prompts = read_prompts(prompt_file)
    end_index = args.end_index if args.end_index is not None else len(prompts)
    shard = prompts[args.start_index:end_index]

    os.makedirs(args.save_path, exist_ok=True)
    if args.condition_on_last_frame:
        anchoring = f"k={k_anchor} anchor block" \
            + (" + first-frame sink" if condition_on_sink_frame else "")
    else:
        anchoring = "none (independent chunks)"
    print(f"[VBench/Wan2.2-TI2V-5B chunk-by-chunk anchored] {len(shard)} prompts "
          f"(indices {args.start_index}:{end_index} of {len(prompts)}) "
          f"x {num_videos} clips x {args.num_chunks} chunks -> {args.save_path}")
    print(f"[VBench] chunk anchoring: {anchoring}")
    if args.condition_on_last_frame:
        new_per_chunk = args.frames_per_chunk - (pinned if args.drop_boundary_frame else 0)
        print(f"[VBench] anchor block k={k_anchor} latents | {n_clean} pinned latents | "
              f"{pinned} pinned pixel frames/chunk -> {new_per_chunk} new frames per chunk "
              f"x {args.num_chunks} chunks = "
              f"{new_per_chunk * (args.num_chunks - 1) + args.frames_per_chunk} frames | "
              f"{anchor_window} frames handed forward")

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
            frames = generate_anchored_video(
                pipe, args, prompt, seed_base, k_anchor,
                condition_on_sink_frame, anchor_window,
            )
            save_video(frames, out_path, fps=args.fps, quality=args.quality)
            print(f"  [{args.start_index + p_i + 1}/{end_index}] clip {index} "
                  f"({len(frames)} frames, seeds {seed_base}..{seed_base + args.num_chunks - 1}) "
                  f"-> {out_path}")


if __name__ == "__main__":
    main()
