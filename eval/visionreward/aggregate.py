"""
Merge VisionReward .jsonl score files into a per-arm summary and an arm comparison.

Each positional argument is one arm, given as `name=path` (or just `path`, whose stem
becomes the name). A path may be a single .jsonl, a glob over shard files, or a
directory containing .jsonl files — sharded runs are merged and de-duplicated by clip
path, keeping the last record for each.

    python eval/visionreward/aggregate.py \
        base=/work/nlp/hzhao/evaluations/visionreward/demos/base.jsonl \
        cbc=/work/nlp/hzhao/evaluations/visionreward/demos/chunk-by-chunk.jsonl \
        ttt=/work/nlp/hzhao/evaluations/visionreward/demos/e2e-ttt-fomaml.jsonl \
        --per_question --csv summary.csv

Every score is an ABSOLUTE, single-video quantity: `mean(answers * weights)` over the
29-question checklist for that one clip and its prompt. No competitor video enters it,
and greedy decoding over deterministic frame sampling makes it reproducible per clip.

Beyond the headline mean it reports:
  * a paired comparison against the first arm — arms are matched on prompt, which
    removes the prompt-difficulty variance that dominates a 100-clip mean. Note this
    is not an extra model call and loses nothing relative to upstream's dedicated
    `compare_two_videos`: that function returns `sum((a1 - a2) * weight) > 0`, and
    since `sum(a*weight) = 29 * mean(a*weight)`, it reduces exactly to `score1 >
    score2`. The `win%` column below therefore IS upstream's pairwise verdict,
    recovered from per-video scores at no extra cost;
  * `--per_question` per-checklist-item yes-rates, which is where the arms actually
    differ (consistency/dynamics items vs. appearance items).
"""

import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visionreward_video import DEFAULT_QUESTIONS_PATH, load_checklist  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Summarize VisionReward score files.")
    p.add_argument("arms", nargs="+",
                   help="One `name=path` per arm (path may be a .jsonl, a glob, or a "
                        "directory). The first arm is the comparison baseline.")
    p.add_argument("--per_question", action="store_true",
                   help="Also print per-checklist-question yes-rates.")
    p.add_argument("--questions_path", type=str, default=DEFAULT_QUESTIONS_PATH)
    p.add_argument("--csv", type=str, default=None,
                   help="Write the per-clip scores (one column per arm) to this CSV.")
    return p.parse_args()


def expand(path):
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "*.jsonl")))
    matches = sorted(glob.glob(path))
    return matches or [path]


def load_arm(path):
    """Load and de-duplicate one arm's records, keyed by clip path."""
    records = {}
    for f in expand(path):
        if not os.path.exists(f):
            raise SystemExit(f"No such score file: {f}")
        with open(f, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # truncated tail of a killed job
                records[r["video"]] = r
    if not records:
        raise SystemExit(f"No records loaded from {path}")
    return records


def mean(xs):
    return sum(xs) / len(xs)


def stderr(xs):
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var / len(xs))


def key_of(record):
    """Pair clips across arms by prompt — arms use different file names for the
    same prompt (VBench `<prompt>-<i>.mp4` vs. custom `<prompt[:30]>/<name>.mp4`)."""
    return record["prompt"]


def main():
    args = parse_args()

    arms = []
    for spec in args.arms:
        name, _, path = spec.partition("=")
        if not path:
            name, path = os.path.splitext(os.path.basename(name))[0], name
        arms.append((name, load_arm(path)))

    width = max(len(n) for n, _ in arms)
    print(f"\n{'arm':<{width}}  {'n':>4}  {'mean':>9}  {'±sem':>7}  {'yes-rate':>8}  trunc")
    print("-" * (width + 45))
    for name, records in arms:
        scores = [r["score"] for r in records.values()]
        yes = mean([r["num_yes"] / r["num_questions"] for r in records.values()])
        trunc = sum(int(r.get("truncated", False)) for r in records.values())
        print(f"{name:<{width}}  {len(scores):>4}  {mean(scores):>+9.4f}  "
              f"{stderr(scores):>7.4f}  {yes:>8.1%}  {trunc:>5}")

    if any(r.get("truncated") for _, recs in arms for r in recs.values()):
        print("\nNOTE: 'trunc' clips were scored on their first 24s only ('chat' frame "
              "sampling). Re-score with --frame_sampling uniform to cover the tail.")

    # Paired comparison — same prompt, arm vs. baseline.
    if len(arms) > 1:
        base_name, base = arms[0]
        base_by_key = {key_of(r): r["score"] for r in base.values()}
        print(f"\nPaired vs. {base_name} (same prompt):")
        # Ties are broken out rather than folded into the loss column: the checklist is
        # 29 binary answers, so two arms agreeing on all of them is common (25% of
        # prompts in the first checkpoint-vs-checkpoint comparison), and counting those
        # as non-wins made a 62.7% win rate read as 47%.
        print(f"{'arm':<{width}}  {'pairs':>5}  {'Δmean':>9}  {'±sem':>7}  "
              f"{'W/T/L':>12}  {'win%*':>6}")
        print("-" * (width + 52))
        for name, records in arms[1:]:
            deltas = [
                r["score"] - base_by_key[key_of(r)]
                for r in records.values() if key_of(r) in base_by_key
            ]
            if not deltas:
                print(f"{name:<{width}}  {'0':>5}   (no shared prompts)")
                continue
            w = sum(d > 0 for d in deltas)
            l = sum(d < 0 for d in deltas)
            t = len(deltas) - w - l
            decided = w + l
            rate = f"{w / decided:>5.1%}" if decided else "    --"
            print(f"{name:<{width}}  {len(deltas):>5}  {mean(deltas):>+9.4f}  "
                  f"{stderr(deltas):>7.4f}  {f'{w}/{t}/{l}':>12}  {rate:>6}")
        print("* win% is over decided prompts only (ties excluded).")

    if args.per_question:
        questions, weights = load_checklist(args.questions_path)
        print(f"\nPer-question yes-rate ({len(questions)} checklist items):")
        header = "  ".join(f"{n:>8.8}" for n, _ in arms)
        print(f"{'w':>7}  {header}  question")
        for qi, question in enumerate(questions):
            rates = []
            for _, records in arms:
                vals = [
                    1.0 if r["answers"][qi] == "yes" else 0.0
                    for r in records.values() if len(r.get("answers", [])) > qi
                ]
                rates.append(mean(vals) if vals else float("nan"))
            cells = "  ".join(f"{v:>8.1%}" for v in rates)
            print(f"{weights[qi]:>7.3f}  {cells}  {question[:70]}")

    if args.csv:
        keys = sorted({key_of(r) for _, recs in arms for r in recs.values()})
        with open(args.csv, "w", newline="") as f:
            import csv

            writer = csv.writer(f)
            writer.writerow(["prompt"] + [n for n, _ in arms])
            for k in keys:
                row = [k]
                for _, records in arms:
                    match = next((r for r in records.values() if key_of(r) == k), None)
                    row.append(f"{match['score']:.6f}" if match else "")
                writer.writerow(row)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
