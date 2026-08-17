#!/usr/bin/env python
"""Summarize a test-time inner-loop trace written by `--ttt_diagnostics`.

The trace answers one question: as the chunk index grows, is the LoRA scratchpad still
*adapting*, or has it drifted past anything meta-training optimized for?

Read the columns in this order:

  probe_chunk0  The scratchpad evaluated against chunk 0's latents, held fixed for the
                whole video, on FIXED (timestep, noise) draws. Nothing in it varies but
                phi, so a rise is unambiguous: the scratchpad is overwriting the start of
                the video. This is the column that matches the reported failure.
  probe_self    Same fixed draws, but against the chunk just generated, measured BEFORE
                its memorize step -- "does phi adapted on chunks 0..k-1 already explain
                chunk k". Confounded by chunk content, so read it against probe_chunk0.
  dW/W          ||scaling * B @ A||_F / ||W_base||_F. The only gauge-invariant magnitude
                (A and B norms individually are invariant to A -> cA, B -> B/c), i.e. how
                hard the adapter is actually pulling on the pretrained weights.
  drift_A/B     ||phi - phi_0|| / ||phi_0||, per group. lora_B is zero-init, so whatever
                meta-training left there is the entire denominator -- B's ratio is the one
                that can pass 1 while the aggregate still looks tame.
  live/mstr     drift as the DiT sees it (bf16 leaves) over drift as the optimizer holds
                it (fp32 master). The write-back quantizes, and bf16's quantum at LoRA's
                parameter scale is comparable to one inner step, so this is < 1. A value
                falling toward 0 means the readback is eating the update -- the failure
                mode that made the pre-master in-place bf16 `sub_` discard most of itself.
  |step|        Displacement realized per optimizer step, measured on the fp32 master.
  gain          step_norm / (lr * grad_norm_post_clip). AdamW's preconditioned step moves
                each coordinate by ~lr no matter how small the gradient got, so a flat
                loss with a climbing gain means the optimizer is walking at full speed on
                noise rather than converging.
  clip%         Fraction of LoRA tensors whose gradient hit max_inner_grad_norm.

Usage:
    python eval/ttt_diagnostics/summarize.py trace.jsonl
    python eval/ttt_diagnostics/summarize.py trace.jsonl --plot trace.png
    python eval/ttt_diagnostics/summarize.py trace.jsonl --video "A red fox trots ..."
"""
import argparse
import json
import os
from collections import defaultdict


def load(path):
    """Group records by video, preserving order. Returns {video: {"step": [...], "chunk": [...]}}.

    Traces open in APPEND mode, so a resubmitted job re-runs the video it died inside and
    appends a second set of records for it. Deduplicate keeping the LAST occurrence, keyed on
    (chunk, when) for chunk records and (chunk, step_in_chunk) for step records — otherwise a
    resumed video shows each early chunk twice."""
    videos = defaultdict(lambda: defaultdict(dict))
    dupes = 0
    with open(path) as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # A run killed mid-write leaves a partial last line; everything before it
                # is still valid, so warn rather than lose the trace.
                print(f"[warn] {path}:{line_no}: malformed JSON, skipping")
                continue
            kind = rec.get("kind")
            key = ((rec.get("chunk"), rec.get("when")) if kind == "chunk"
                   else (rec.get("chunk"), rec.get("step_in_chunk")))
            bucket = videos[rec.get("video")][kind]
            if key in bucket:
                dupes += 1
            bucket[key] = rec
    if dupes:
        print(f"[note] {path}: dropped {dupes} superseded record(s) from a resumed run")
    return {v: {k: list(b.values()) for k, b in kinds.items()} for v, kinds in videos.items()}


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _fmt(value, spec=".4f"):
    return "-" if value is None else format(value, spec)


def per_chunk_rows(records):
    """One row per chunk: the chunk-level probes joined with the mean of that chunk's steps."""
    steps_by_chunk = defaultdict(list)
    for rec in records.get("step", []):
        steps_by_chunk[rec["chunk"]].append(rec)

    rows = []
    for chunk_rec in records.get("chunk", []):
        chunk = chunk_rec["chunk"]
        steps = steps_by_chunk.get(chunk, [])
        # Drift is cumulative, so the chunk's state is its LAST step; the per-step
        # magnitudes are averaged over the chunk's steps.
        last = steps[-1] if steps else {}
        rows.append({
            "chunk": chunk,
            "probe_self": chunk_rec.get("probe_self"),
            "probe_chunk0": chunk_rec.get("probe_chunk0"),
            "delta_w_ratio": chunk_rec.get("delta_w_ratio"),
            "drift_ratio_A": last.get("drift_ratio_A"),
            "drift_ratio_B": last.get("drift_ratio_B"),
            "drift_norm": last.get("drift_norm"),
            "live_over_master": last.get("live_over_master"),
            "step_norm": _mean([s.get("step_norm") for s in steps]),
            "precond_gain": _mean([s.get("precond_gain") for s in steps]),
            "grad_norm": _mean([s.get("grad_norm_pre_clip") for s in steps]),
            "loss": _mean([s.get("loss") for s in steps]),
            "clip_frac": _mean([
                s["num_tensors_clipped"] / s["num_tensors"]
                for s in steps if s.get("num_tensors")
            ]),
            "num_steps": len(steps),
        })
    return rows


def report(video, records):
    rows = per_chunk_rows(records)
    if not rows:
        print(f"(no chunk records for {video!r})")
        return rows

    total_steps = len(records.get("step", []))
    print(f"\n=== {video} ===")
    print(f"{total_steps} optimizer steps over {len(rows)} chunks "
          f"(no phi_0 reset within a video)\n")

    header = (f"{'chunk':>5} {'probe_self':>11} {'probe_ch0':>11} {'dW/W':>9} "
              f"{'drift_A':>9} {'drift_B':>9} {'live/mstr':>9} {'|step|':>10} {'gain':>9} "
              f"{'|grad|':>9} {'clip%':>6}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['chunk']:>5} {_fmt(r['probe_self']):>11} {_fmt(r['probe_chunk0']):>11} "
              f"{_fmt(r['delta_w_ratio'], '.5f'):>9} {_fmt(r['drift_ratio_A'], '.5f'):>9} "
              f"{_fmt(r['drift_ratio_B'], '.5f'):>9} {_fmt(r['live_over_master'], '.3f'):>9} "
              f"{_fmt(r['step_norm'], '.3e'):>10} "
              f"{_fmt(r['precond_gain'], '.2f'):>9} {_fmt(r['grad_norm'], '.3e'):>9} "
              f"{_fmt(r['clip_frac'] * 100 if r['clip_frac'] is not None else None, '.0f'):>6}")

    # First-to-last deltas on the columns that decide the diagnosis.
    first, last = rows[0], rows[-1]
    print()
    for label, key in (("probe_chunk0", "probe_chunk0"), ("probe_self", "probe_self"),
                       ("dW/W", "delta_w_ratio"), ("drift_B", "drift_ratio_B"),
                       ("precond_gain", "precond_gain")):
        a, b = first.get(key), last.get(key)
        if a is None or b is None:
            continue
        rel = f" ({b / a:.2f}x)" if a > 0 else ""
        print(f"  {label:<13} chunk {first['chunk']} -> {last['chunk']}: "
              f"{a:.5g} -> {b:.5g}{rel}")

    # The self-check the probe design gives for free.
    if rows[0].get("probe_self") is not None and rows[0].get("probe_chunk0") is not None:
        if abs(rows[0]["probe_self"] - rows[0]["probe_chunk0"]) > 1e-9:
            print("\n  [warn] probe_self != probe_chunk0 at chunk 0, but the reference IS "
                  "chunk 0 -- the probe is not deterministic.")
    return rows


def plot(all_rows, out_path):
    try:
        import matplotlib
    except ModuleNotFoundError:
        # The table above is pure stdlib and runs in `diffsynth`; only the plot needs
        # matplotlib, which on this box lives in the `data` env.
        print("\n[skip] --plot needs matplotlib, which is not in this interpreter. "
              "Re-run the summarizer under an env that has it, e.g.:\n"
              f"  /work/nlp/hzhao/miniforge3/envs/data/bin/python {__file__} <trace> --plot {out_path}")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("Fixed-probe memorize loss", [("probe_chunk0", "vs chunk 0 (fixed)"),
                                       ("probe_self", "vs current chunk")]),
        ("Effective adapter magnitude", [("delta_w_ratio", "||B@A|| / ||W_base||")]),
        ("Drift from phi_0", [("drift_ratio_A", "lora_A"), ("drift_ratio_B", "lora_B")]),
        ("Realized step norm", [("step_norm", "||phi_t - phi_t-1||")]),
        ("AdamW preconditioner gain", [("precond_gain", "|step| / (lr*|grad|)")]),
        ("Memorize loss (random sigma)", [("loss", "as logged")]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (title, series) in zip(axes.flat, panels):
        for video, rows in all_rows.items():
            for key, label in series:
                xs = [r["chunk"] for r in rows if r.get(key) is not None]
                ys = [r[key] for r in rows if r.get(key) is not None]
                if xs:
                    suffix = f" [{video[:18]}]" if len(all_rows) > 1 else ""
                    ax.plot(xs, ys, marker="o", ms=3, label=label + suffix)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("chunk index")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    # Log scale where the quantity is expected to span orders of magnitude.
    axes.flat[3].set_yscale("log")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"\nWrote {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", help="JSONL written by --ttt_diagnostics.")
    parser.add_argument("--video", default=None,
                        help="Report only the video whose prompt starts with this string "
                             "(default: every video in the trace).")
    parser.add_argument("--plot", default=None, help="Also write a 6-panel PNG here.")
    args = parser.parse_args()

    videos = load(args.trace)
    if args.video:
        videos = {k: v for k, v in videos.items() if k and k.startswith(args.video)}
        if not videos:
            parser.error(f"no video in {args.trace} starts with {args.video!r}")

    all_rows = {}
    for video, records in videos.items():
        rows = report(video, records)
        if rows:
            all_rows[video or "(unnamed)"] = rows

    if args.plot and all_rows:
        plot(all_rows, args.plot)


if __name__ == "__main__":
    main()
