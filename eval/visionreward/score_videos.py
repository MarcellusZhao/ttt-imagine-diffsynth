"""
Score a directory of generated clips with VisionReward-Video.

Every clip needs the prompt it was generated from (three of the 29 checklist
questions interpolate it), so the driver has to recover prompt-per-file from the
layout the sampler wrote. Two layouts cover everything this repo produces, plus an
explicit escape hatch:

  --layout vbench   <videos_path>/<prompt>-<index>.mp4
                    The VBench protocol used by examples/wanvideo/model_inference/
                    vbench/*. Prompt is the filename minus the trailing -<index>.
                    Only usable for prompts short enough to be a filename.

  --layout custom   <videos_path>/<prompt[:30]>/<name>.mp4
                    What the custom-prompts/* drivers write. Prompts are truncated to
                    30 chars in the directory name, so the FULL prompt is recovered by
                    matching that prefix against --prompt_file (required). This is the
                    layout for the 100 Causal-Forcing demo prompts, which run to 775
                    characters and cannot be filenames.

  --layout map      --prompt_map is a JSON object {relative/path.mp4: "prompt"}.

Output is one JSON line per clip in <output>, written incrementally, so a re-run with
--skip_existing resumes and a killed job loses at most one clip. Shard across GPUs
with --shard_index/--num_shards (round-robin over the file list; every shard appends
to its own .jsonl, and aggregate.py merges them).

Example — score one arm's 100-demo-prompt run:

    python eval/visionreward/score_videos.py \
        --videos_path /work/nlp/hzhao/evaluations/visionreward/demos/e2e-ttt-fomaml \
        --layout custom \
        --prompt_file eval/visionreward/prompts/causal_forcing_demos.txt \
        --frame_sampling uniform \
        --output /work/nlp/hzhao/evaluations/visionreward/demos/e2e-ttt-fomaml.jsonl
"""

import argparse
import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visionreward_video import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    DEFAULT_QUESTIONS_PATH,
    DEFAULT_WEIGHT_PATH,
    NUM_FRAMES,
    VisionRewardVideo,
)

VBENCH_NAME_RE = re.compile(r"^(?P<prompt>.+)-(?P<index>\d+)$")
PROMPT_DIR_TRUNCATION = 30  # custom-prompts/* use prompt[:30] as the directory name


def parse_args():
    p = argparse.ArgumentParser(
        description="Score generated videos with VisionReward-Video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--videos_path", type=str, required=True,
                   help="Directory of clips to score.")
    p.add_argument("--layout", type=str, default="vbench",
                   choices=["vbench", "custom", "map"],
                   help="How to recover each clip's prompt (see module docstring).")
    p.add_argument("--prompt_file", type=str, default=None,
                   help="One prompt per line; required for --layout custom.")
    p.add_argument("--prompt_map", type=str, default=None,
                   help="JSON {relpath: prompt}; required for --layout map.")
    p.add_argument("--pattern", type=str, default=None,
                   help="Glob for clips, relative to --videos_path. Defaults to "
                        "'*.mp4' for vbench and '*/*.mp4' for custom.")
    p.add_argument("--output", type=str, required=True,
                   help="Output .jsonl (one record per clip, written incrementally).")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip clips already present in --output (resumable).")

    p.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                   help="VisionReward-Video weights (local dir or HF repo id).")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--questions_path", type=str, default=DEFAULT_QUESTIONS_PATH)
    p.add_argument("--weight_path", type=str, default=DEFAULT_WEIGHT_PATH)

    p.add_argument("--frame_sampling", type=str, default="chat",
                   choices=["chat", "uniform"],
                   help="'chat' (upstream) takes one frame per second and therefore "
                        "sees only the first 24s of a clip; 'uniform' spreads 24 "
                        "frames over the whole clip. Use 'uniform' for long videos, "
                        "and keep the choice identical across arms.")
    p.add_argument("--num_frames", type=int, default=NUM_FRAMES,
                   help="Frames shown to the model. 24 is what VisionReward was "
                        "trained with; each frame costs 66 tokens against the "
                        "backbone's 2048-token window, so raising this is unsafe.")
    p.add_argument("--batch_size", type=int, default=1,
                   help="Checklist questions per forward pass. >1 is faster but "
                        "exercises the padded batch path — validate it once with "
                        "--verify_batching.")
    p.add_argument("--verify_batching", action="store_true",
                   help="Before scoring, check that --batch_size reproduces the "
                        "batch-1 answers on the first clip, then exit if it does not.")

    p.add_argument("--num_shards", type=int, default=1,
                   help="Split the clip list over N processes (round-robin).")
    p.add_argument("--shard_index", type=int, default=0,
                   help="Which shard this process handles, in [0, num_shards).")
    p.add_argument("--limit", type=int, default=None,
                   help="Score at most this many clips (smoke tests).")
    return p.parse_args()


def read_prompts(path):
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]


def resolve_prompts(args, files):
    """Map each clip path to the prompt it was generated from."""
    if args.layout == "vbench":
        out = {}
        for path in files:
            stem = os.path.splitext(os.path.basename(path))[0]
            m = VBENCH_NAME_RE.match(stem)
            # A clip with no -<index> suffix is still scoreable: the whole stem is
            # the prompt. Only the index is optional, never the prompt.
            out[path] = m.group("prompt") if m else stem
        return out

    if args.layout == "custom":
        if not args.prompt_file:
            raise SystemExit("--layout custom requires --prompt_file to recover the "
                             "full prompt from the truncated directory name.")
        prompts = read_prompts(args.prompt_file)
        by_prefix = {}
        for prompt in prompts:
            by_prefix.setdefault(prompt[:PROMPT_DIR_TRUNCATION], []).append(prompt)
        collisions = {k: v for k, v in by_prefix.items() if len(v) > 1}
        if collisions:
            raise SystemExit(
                f"{len(collisions)} prompt(s) in {args.prompt_file} share their first "
                f"{PROMPT_DIR_TRUNCATION} characters, so the sampler's output "
                f"directories are ambiguous. First: {next(iter(collisions))!r}"
            )

        out, unmatched = {}, []
        for path in files:
            key = os.path.basename(os.path.dirname(path))
            if key in by_prefix:
                out[path] = by_prefix[key][0]
            else:
                unmatched.append(path)
        if unmatched:
            raise SystemExit(
                f"{len(unmatched)} clip(s) sit in directories not matching any prompt "
                f"in {args.prompt_file} — wrong prompt file for this run?\n  "
                + "\n  ".join(unmatched[:5])
            )
        return out

    if not args.prompt_map:
        raise SystemExit("--layout map requires --prompt_map.")
    with open(args.prompt_map, "r") as f:
        mapping = json.load(f)
    out, missing = {}, []
    for path in files:
        rel = os.path.relpath(path, args.videos_path)
        if rel in mapping:
            out[path] = mapping[rel]
        else:
            missing.append(rel)
    if missing:
        raise SystemExit(f"{len(missing)} clip(s) absent from --prompt_map, e.g. {missing[:3]}")
    return out


def already_scored(output_path):
    """Clip paths already recorded in a previous (possibly interrupted) run."""
    if not os.path.exists(output_path):
        return set()
    done = set()
    with open(output_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["video"])
            except (json.JSONDecodeError, KeyError):
                # A truncated final line from a killed job — that clip is re-scored.
                continue
    return done


def main():
    args = parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit(f"--shard_index {args.shard_index} outside [0, {args.num_shards})")

    pattern = args.pattern or ("*/*.mp4" if args.layout == "custom" else "*.mp4")
    files = sorted(glob.glob(os.path.join(args.videos_path, pattern)))
    if not files:
        raise SystemExit(f"No clips matching '{pattern}' under {args.videos_path}")

    prompts = resolve_prompts(args, files)
    files = files[args.shard_index::args.num_shards]
    if args.limit is not None:
        files = files[:args.limit]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    done = already_scored(args.output) if args.skip_existing else set()
    todo = [f for f in files if f not in done]

    print(f"[VisionReward] {len(todo)} clip(s) to score "
          f"(shard {args.shard_index}/{args.num_shards}, {len(done)} already done)")
    print(f"[VisionReward] layout={args.layout} frame_sampling={args.frame_sampling} "
          f"batch_size={args.batch_size}")
    print(f"[VisionReward] -> {args.output}")
    if not todo:
        return

    scorer = VisionRewardVideo(
        model_path=args.model_path, device=args.device,
        questions_path=args.questions_path, weight_path=args.weight_path,
        frame_sampling=args.frame_sampling, num_frames=args.num_frames,
    )

    if args.verify_batching and args.batch_size > 1:
        verify_batching(scorer, todo[0], prompts[todo[0]], args.batch_size)

    scores, truncated, t0 = [], 0, time.time()
    with open(args.output, "a") as out:
        for i, path in enumerate(todo, 1):
            result = scorer.score_video(path, prompts[path], batch_size=args.batch_size)
            record = {"video": path, "prompt": prompts[path], **result}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

            scores.append(result["score"])
            truncated += int(result["truncated"])
            elapsed = time.time() - t0
            print(f"  [{i}/{len(todo)}] score={result['score']:+.4f} "
                  f"yes={result['num_yes']}/{result['num_questions']} "
                  f"({elapsed / i:.1f}s/clip) {os.path.basename(path)}")

    mean = sum(scores) / len(scores)
    print(f"\n[VisionReward] {len(scores)} clip(s), mean score {mean:+.4f}, "
          f"{time.time() - t0:.0f}s total")
    if truncated:
        print(f"[VisionReward] WARNING: {truncated}/{len(scores)} clip(s) run longer "
              f"than {args.num_frames}s and 'chat' sampling stopped at their first "
              f"{args.num_frames}s — the tail was never scored. Re-run with "
              f"--frame_sampling uniform to cover the whole clip.")


def verify_batching(scorer, video_path, prompt, batch_size):
    """Guard the padded-batch path against the reference batch-1 path."""
    print(f"[VisionReward] verifying batch_size={batch_size} against batch_size=1 "
          f"on {os.path.basename(video_path)} …")
    ref = scorer.score_video(video_path, prompt, batch_size=1)
    got = scorer.score_video(video_path, prompt, batch_size=batch_size)
    if ref["answers"] != got["answers"]:
        diff = [i for i, (a, b) in enumerate(zip(ref["answers"], got["answers"])) if a != b]
        raise SystemExit(
            f"batch_size={batch_size} disagrees with batch_size=1 on "
            f"{len(diff)} of {len(ref['answers'])} question(s) (indices {diff[:10]}). "
            f"Re-run with --batch_size 1."
        )
    print(f"[VisionReward] batching verified (score {ref['score']:+.4f}).")


if __name__ == "__main__":
    main()
