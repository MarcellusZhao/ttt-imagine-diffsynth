#!/usr/bin/env python3
"""Merge per-dataset train_diffsynth.csv files into a single CSV.

Each dataset folder under the base directory contains a `train_diffsynth.csv`
with columns `video,prompt`. This script concatenates the rows from the
specified datasets into one output CSV.
"""
import argparse
import csv
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="List of dataset folder names to merge (e.g. bdd100k ego4d panda).",
    )
    parser.add_argument(
        "--base-dir",
        default="/work/nlp/hzhao/datasets/e2e-ttt-video",
        help="Base directory containing the dataset folders.",
    )
    parser.add_argument(
        "--csv-name",
        default="train_diffsynth.csv",
        help="Name of the per-dataset CSV file to merge.",
    )
    parser.add_argument(
        "--output-name",
        default="train_diffsynth_merged.csv",
        help="Name of the output CSV file.",
    )
    parser.add_argument(
        "--data-size",
        default=10000,
        type=int,
        help="Size of the dataset.",
    )
    args = parser.parse_args()

    output = os.path.join(args.base_dir, args.output_name)

    total_rows = 0
    with open(output, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["video", "prompt"])
        data_size = args.data_size
        overall_count = 0
        print(f"Merging {data_size} rows from {len(args.datasets)} dataset(s) -> {output}")
        for dataset in args.datasets:
            csv_path = os.path.join(args.base_dir, dataset, args.csv_name)
            if not os.path.isfile(csv_path):
                print(f"[WARN] skipping {dataset}: {csv_path} not found")
                continue

            with open(csv_path, newline="") as fin:
                reader = csv.reader(fin)
                header = next(reader, None)  # skip header row
                count = 0
                for row in reader:
                    if not row:  # skip blank trailing lines
                        continue
                    writer.writerow(row)
                    count += 1
                    overall_count += 1
                    if overall_count >= data_size:
                        break
            print(f"[OK] {dataset}: {count} rows")
            total_rows += count

    print(f"\nMerged {total_rows} rows from {len(args.datasets)} dataset(s) -> {output}")


if __name__ == "__main__":
    main()
