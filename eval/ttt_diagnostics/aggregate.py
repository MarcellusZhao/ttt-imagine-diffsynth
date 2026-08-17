#!/usr/bin/env python
"""Aggregate several `--ttt_diagnostics` traces and plot the probe curves against chunk index.

A single video's probes are noisy enough to point the wrong way -- on one prompt `probe_self`
FELL across 24 chunks while the mean over 29 prompts rose -- so read the aggregate, not one
clip. Each trace is one arm/depth; they are overlaid, with a shaded +-1 sem band.

The three panels, and what a rise in each means:

  probe_chunk0   current phi vs chunk 0's latents, held fixed for the whole video, on fixed
                 (timestep, noise) draws. Nothing varies but phi -> a rise is unambiguous
                 forgetting of the video's start.
  probe_self     same draws vs the chunk just generated, taken BEFORE its memorize step ->
                 a rise is the meta-objective's generalization decaying with depth.
  dW/W           ||scaling * B @ A|| / ||W_base||, the gauge-invariant adapter magnitude ->
                 a rise is the adapter pulling harder on the base weights.

Usage:
    python eval/ttt_diagnostics/aggregate.py results/ttt-diag/**/*.jsonl -o out.png
    python eval/ttt_diagnostics/aggregate.py a.jsonl b.jsonl --label 24-chunk --label 48-chunk

Needs matplotlib for -o; the table always prints. On this box matplotlib lives in the `data`
env (/work/nlp/hzhao/miniforge3/envs/data/bin/python), not `diffsynth`.
"""
import argparse
import collections
import json
import os
import statistics as st

KEYS = ("probe_chunk0", "probe_self", "delta_w_ratio")
TITLES = {
    "probe_chunk0": "probe_chunk0 — fit to chunk 0 (fixed)\nRISE = forgetting the video's start",
    "probe_self": "probe_self — fit to the NEXT chunk\nRISE = generalization decaying",
    "delta_w_ratio": "dW/W — effective adapter magnitude",
}


def load(path):
    """{chunk_index: {key: [value per video]}} from one trace.

    Traces are opened in APPEND mode, so a job killed at its wall clock and then resubmitted
    re-runs the video it died inside (the driver's --skip_existing keys on the output video,
    which that video never got) and appends a second set of records for it. Deduplicate on
    (video, chunk, when), keeping the LAST occurrence, or the re-run video's early chunks are
    counted twice and skew the mean toward one clip."""
    latest = {}
    order = []
    dupes = 0
    for line_no, line in enumerate(open(path), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            # A job killed mid-write leaves a partial last line; the rest is valid.
            print(f"[warn] {path}:{line_no}: malformed JSON, skipping")
            continue
        if rec.get("kind") != "chunk":
            continue
        key = (rec.get("video"), rec["chunk"], rec.get("when"))
        if key in latest:
            dupes += 1
        else:
            order.append(key)
        latest[key] = rec
    if dupes:
        print(f"[note] {os.path.basename(path)}: dropped {dupes} superseded record(s) "
              f"from a resumed run")
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for key in order:
        rec = latest[key]
        for k in KEYS:
            if rec.get(k) is not None:
                by[rec["chunk"]][k].append(rec[k])
    return by


def table(label, by):
    chunks = sorted(by)
    if not chunks:
        print(f"=== {label}: no chunk records ===\n")
        return
    n = len(by[chunks[0]].get("probe_chunk0", []))
    print(f"=== {label} | {len(chunks)} chunks | {n} videos at chunk 0 ===")
    print(f"{'chunk':>5} {'probe_ch0':>10} {'±sem':>8} {'probe_self':>11} {'dW/W':>9} {'n':>4}")
    step = max(1, len(chunks) // 24)
    for c in chunks[::step]:
        v = by[c].get("probe_chunk0", [])
        s = by[c].get("probe_self", [])
        d = by[c].get("delta_w_ratio", [])
        sem = (st.stdev(v) / len(v) ** 0.5) if len(v) > 1 else 0.0
        print(f"{c:>5} {st.mean(v):>10.4f} {sem:>8.4f} {st.mean(s):>11.4f} "
              f"{st.mean(d):>9.5f} {len(v):>4}")
    first, last = chunks[0], chunks[-1]
    for key in KEYS:
        a, b = by[first].get(key), by[last].get(key)
        if a and b:
            ma, mb = st.mean(a), st.mean(b)
            print(f"  {key:<14} {first} -> {last}: {ma:.4f} -> {mb:.4f}"
                  + (f"  ({mb / ma:.2f}x)" if ma else ""))
    print()


def plot(series, out_path):
    try:
        import matplotlib
    except ModuleNotFoundError:
        print(f"[skip] -o needs matplotlib. Try:\n"
              f"  /work/nlp/hzhao/miniforge3/envs/data/bin/python {__file__} ... -o {out_path}")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for (label, by), color in zip(series.items(),
                                  ("tab:blue", "tab:red", "tab:green", "tab:orange")):
        xs = sorted(by)
        for ax, key in zip(axes, KEYS):
            pts = [(c, by[c][key]) for c in xs if by[c].get(key)]
            if not pts:
                continue
            cx = [c for c, _ in pts]
            m = [st.mean(v) for _, v in pts]
            ax.plot(cx, m, color=color, lw=2,
                    label=f"{label} (n={len(pts[0][1])})")
            if key != "delta_w_ratio":
                sem = [(st.stdev(v) / len(v) ** 0.5) if len(v) > 1 else 0.0 for _, v in pts]
                ax.fill_between(cx, [a - b for a, b in zip(m, sem)],
                                [a + b for a, b in zip(m, sem)], color=color, alpha=0.2)
            ax.set_title(TITLES[key], fontsize=10)
            ax.set_xlabel("chunk index")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    fig.suptitle("E2E-TTT test-time scratchpad, mean over prompts (shaded = ±1 sem)", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traces", nargs="+", help="JSONL traces, one per arm/depth.")
    parser.add_argument("--label", action="append", default=None,
                        help="Legend label per trace, in order (default: parent dir name).")
    parser.add_argument("-o", "--out", default=None, help="Write the 3-panel PNG here.")
    args = parser.parse_args()

    if args.label and len(args.label) != len(args.traces):
        parser.error(f"got {len(args.label)} --label for {len(args.traces)} traces")

    series = {}
    for i, path in enumerate(args.traces):
        label = args.label[i] if args.label else os.path.basename(os.path.dirname(path))
        by = load(path)
        table(label, by)
        if by:
            series[label] = by

    if args.out and series:
        plot(series, args.out)


if __name__ == "__main__":
    main()
