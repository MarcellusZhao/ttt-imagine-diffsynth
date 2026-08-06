#!/usr/bin/env python
"""Add a `num_frames` column to a `video,prompt` training CSV so
`--length_grouped_batching` can order clips by length.

The column is a SCHEDULING HINT only: it decides which clips share an optimizer step,
never what is trained (`LoadVideo` still measures the true frame count at load time). So an
approximate value is fine -- a stale one degrades load balance and nothing else.

Two sources:

  metadata (default, no video I/O)
      Join the source UltraVideo metadata (`long.csv`: clip_id, fps, total_frames, duration)
      on `video == clip_id` and take `round(duration * --fps)`. Use `--fps 16` for the
      curated clips that were resampled to 16 fps offline (the `rsfps16` runs). Duration is
      used rather than `total_frames` because `total_frames` counts frames at the clip's
      SOURCE fps (23.976, 25, ...), not the on-disk rate after resampling.

  probe (`--probe`)
      ffprobe each file under --base-path for its real frame count. Slower (a few minutes
      for 16k clips) but exact, and the right choice if you are unsure the clips were
      actually resampled to a uniform fps.

Examples
--------
    # fast path: derive from the source metadata, assuming 16 fps on disk
    python datasets/ultravideo_long/add_length_column.py \
        --csv datasets/ultravideo_long/ultravideo_long_filtered_all_16k.csv \
        --metadata datasets/ultravideo_long/long.csv --fps 16

    # exact path: ask the files themselves
    python datasets/ultravideo_long/add_length_column.py \
        --csv datasets/ultravideo_long/ultravideo_long_filtered_all_16k.csv --probe \
        --base-path /work/nlp/hzhao/datasets/e2e-ttt-video/ultravideo_long/curated_videos

Writes `<csv stem>_len.csv` next to the input unless --out is given; point the training
config's `dataset_metadata_path` at that file. Extra columns are inert to UnifiedDataset,
which only applies operators to `--data_file_keys` and reads declared `--extra_inputs`.
"""

import argparse, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

import pandas


def probe_num_frames(path):
    """Real decodable frame count. `nb_frames` is absent in some containers, so fall back
    to counting packets, then to duration * avg_frame_rate."""
    def _ffprobe(entries):
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", entries, "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True,
        )
        return [l.strip() for l in out.stdout.splitlines() if l.strip()]

    vals = _ffprobe("stream=nb_frames")
    if vals and vals[0].isdigit() and int(vals[0]) > 0:
        return int(vals[0])
    vals = _ffprobe("stream=nb_read_packets")  # needs -count_packets
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True,
    )
    v = out.stdout.strip()
    if v.isdigit() and int(v) > 0:
        return int(v)
    dur = _ffprobe("stream=duration") or _ffprobe("format=duration")
    rate = _ffprobe("stream=avg_frame_rate")
    try:
        num, den = rate[0].split("/")
        return int(round(float(dur[0]) * float(num) / float(den)))
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="Training CSV with a `video` column.")
    p.add_argument("--out", default=None, help="Output CSV (default: <stem>_len.csv).")
    p.add_argument("--column", default="num_frames", help="Name of the column to add.")
    p.add_argument("--metadata", default=None, help="Source metadata CSV to join (e.g. long.csv).")
    p.add_argument("--metadata-key", default="clip_id", help="Join key in --metadata.")
    p.add_argument("--fps", type=float, default=16.0,
                   help="On-disk frame rate, used to convert duration -> frames in metadata mode.")
    p.add_argument("--probe", action="store_true", help="ffprobe the real files instead of joining metadata.")
    p.add_argument("--base-path", default=None, help="Video directory (required with --probe).")
    p.add_argument("--workers", type=int, default=16, help="Parallel ffprobe workers.")
    args = p.parse_args()

    df = pandas.read_csv(args.csv)
    if "video" not in df.columns:
        sys.exit(f"{args.csv} has no `video` column (found: {list(df.columns)})")

    if args.probe:
        if not args.base_path:
            sys.exit("--probe requires --base-path")
        paths = [os.path.join(args.base_path, v) for v in df["video"]]
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            lengths = list(ex.map(probe_num_frames, paths))
        source = f"ffprobe under {args.base_path}"
    else:
        if not args.metadata:
            sys.exit("give --metadata (fast join) or --probe (exact)")
        meta = pandas.read_csv(args.metadata)
        for col in (args.metadata_key, "duration"):
            if col not in meta.columns:
                sys.exit(f"{args.metadata} has no `{col}` column (found: {list(meta.columns)})")
        dur = dict(zip(meta[args.metadata_key].astype(str), meta["duration"]))
        lengths = [int(round(dur[v] * args.fps)) if v in dur and dur[v] == dur[v] else None
                   for v in df["video"].astype(str)]
        source = f"{args.metadata} duration x {args.fps} fps"

    n_missing = sum(1 for x in lengths if x is None)
    if n_missing:
        # Median keeps unresolved clips in the middle of the ordering: they neither poison a
        # short group nor get quarantined into an all-long one.
        known = sorted(x for x in lengths if x is not None)
        if not known:
            sys.exit("no lengths could be resolved -- check the join key / --base-path")
        fill = known[len(known) // 2]
        print(f"WARNING: {n_missing}/{len(lengths)} clips unresolved; filling with median {fill}")
        lengths = [fill if x is None else x for x in lengths]

    df[args.column] = lengths
    out = args.out or f"{os.path.splitext(args.csv)[0]}_len.csv"
    df.to_csv(out, index=False)

    s = pandas.Series(lengths)
    print(f"[add_length_column] {len(df)} rows | source: {source}")
    print(f"[add_length_column] {args.column}: min {s.min()} p25 {s.quantile(.25):.0f} "
          f"p50 {s.median():.0f} p75 {s.quantile(.75):.0f} p95 {s.quantile(.95):.0f} max {s.max()}")
    print(f"[add_length_column] wrote {out}")
    print(f"[add_length_column] point dataset_metadata_path at it and set "
          f"length_grouped_batching: true")


if __name__ == "__main__":
    main()
