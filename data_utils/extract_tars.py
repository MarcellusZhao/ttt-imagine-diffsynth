"""Extract every tar/zip archive under a folder, one by one.

Handles plain `.tar` plus the compressed variants (`.tar.gz`/`.tgz`,
`.tar.bz2`/`.tbz2`, `.tar.xz`/`.txz`) — `tarfile.open(..., "r:*")` auto-detects
the compression from the file header.  Also handles `.zip` archives; zip files
are always removed after a successful extraction.

Usage
-----
    python extract_tars.py --src /path/to/folder
    python extract_tars.py --src /path/to/folder --dest /path/to/output
    python extract_tars.py --src /path/to/folder --recursive --remove-after

By default each archive is extracted into a sibling directory named after the
archive's stem (e.g. `foo.tar` -> `foo/`). With `--dest`, archives are extracted
under `<dest>/<stem>/` instead.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Iterator

TAR_SUFFIXES = (
    ".tar",
    ".tar.gz", ".tgz",
)

ZIP_SUFFIXES = (".zip",)

# Matches split-archive numeric suffixes: .000  .001  .0000  etc.
_SPLIT_RE = re.compile(r"\.\d{3,}$")

VIDEO_SUFFIXES = (
    ".mp4", ".avi", ".mov", ".mkv", ".webm",
    ".m4v", ".flv", ".wmv", ".mpeg", ".mpg",
)


def iter_archive_files(src: Path, recursive: bool) -> Iterator[Path]:
    """Yield tar/zip archives under `src`, sorted for deterministic order."""
    walker = src.rglob("*") if recursive else src.iterdir()
    matches = [p for p in walker if p.is_file() and _has_archive_suffix(p) and not _is_split_chunk(p)]
    matches.sort()
    yield from matches


def _has_archive_suffix(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(s) for s in TAR_SUFFIXES + ZIP_SUFFIXES)


def _is_split_chunk(path: Path) -> bool:
    """True for files like foo.tar.gz.000 — the base name must itself be a known archive."""
    base = _SPLIT_RE.sub("", path.name)
    return base != path.name and _has_archive_suffix(Path(base))


def _split_base(path: Path) -> Path:
    """Return the logical reassembled-archive path for a split chunk."""
    return path.parent / _SPLIT_RE.sub("", path.name)


def _find_split_groups(src: Path, recursive: bool) -> dict[Path, list[Path]]:
    """Return {base_archive_path: [sorted chunk paths]} for every split group found."""
    walker = src.rglob("*") if recursive else src.iterdir()
    groups: dict[Path, list[Path]] = {}
    for p in walker:
        if p.is_file() and _is_split_chunk(p):
            groups.setdefault(_split_base(p), []).append(p)
    for chunks in groups.values():
        chunks.sort()
    return groups


def reassemble_chunks(chunks: list[Path], dest: Path) -> None:
    """Concatenate sorted split chunks into a single file at dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as out:
        for chunk in chunks:
            with open(chunk, "rb") as f:
                shutil.copyfileobj(f, out)


def _is_zip(path: Path) -> bool:
    return path.name.lower().endswith(ZIP_SUFFIXES)


def _strip_archive_suffix(path: Path) -> str:
    """Return archive stem with the longest matching suffix removed."""
    name = path.name
    lower = name.lower()
    for suffix in sorted(TAR_SUFFIXES + ZIP_SUFFIXES, key=len, reverse=True):
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def extract_one(archive: Path, out_dir: Path) -> None:
    """Extract `archive` into `out_dir`, creating the directory if needed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tar:
        # `filter="data"` (Python 3.12+) blocks unsafe entries: absolute paths,
        # `..` traversal, device files, links pointing outside the dest dir.
        if sys.version_info >= (3, 12):
            tar.extractall(out_dir, filter="data")
        else:
            tar.extractall(out_dir)


def extract_zip(archive: Path, out_dir: Path) -> None:
    """Extract a zip `archive` into `out_dir`, creating the directory if needed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(out_dir)


def _count_videos_in_archive(archive: Path) -> int:
    """Count video files listed inside an archive without fully extracting it."""
    try:
        if _is_zip(archive):
            with zipfile.ZipFile(archive, "r") as zf:
                return sum(
                    1 for name in zf.namelist()
                    if not name.endswith("/") and name.lower().endswith(VIDEO_SUFFIXES)
                )
        else:
            with tarfile.open(archive, "r:*") as tar:
                return sum(
                    1 for m in tar.getmembers()
                    if m.isfile() and m.name.lower().endswith(VIDEO_SUFFIXES)
                )
    except (tarfile.TarError, zipfile.BadZipFile, OSError):
        return 0


def _count_videos_in_dir(path: Path) -> int:
    """Recursively count video files under `path`."""
    return sum(
        1 for p in path.rglob("*")
        if p.is_file() and p.name.lower().endswith(VIDEO_SUFFIXES)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, required=True,
                        help="Folder containing tar archives.")
    parser.add_argument("--dest", type=Path, default=None,
                        help="Output root. Defaults to each archive's parent folder.")
    parser.add_argument("--recursive", action="store_true",
                        help="Recurse into subdirectories of --src.")
    parser.add_argument("--remove-after", action="store_true",
                        help="Delete each archive after a successful extract.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip archives whose output dir already exists and is non-empty.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without extracting or deleting anything.")
    args = parser.parse_args()

    if args.dry_run:
        print("[dry run] no files will be extracted or deleted")

    src: Path = args.src
    if not src.is_dir():
        print(f"error: --src is not a directory: {src}", file=sys.stderr)
        return 2

    failures: list[tuple[Path, Exception]] = []
    total_videos_extracted = 0

    # --- Split archives (must be reassembled before extraction) ---
    split_groups = _find_split_groups(src, args.recursive)
    if split_groups:
        print(f"found {len(split_groups)} split archive group(s) to reassemble")
        for si, (base_path, chunks) in enumerate(sorted(split_groups.items()), start=1):
            stem = _strip_archive_suffix(base_path)
            out_dir = (args.dest / stem) if args.dest else (base_path.parent / stem)
            label = f"[split {si}/{len(split_groups)}]"

            if args.skip_existing and out_dir.is_dir() and any(out_dir.iterdir()):
                print(f"{label} skip (exists): {base_path.name}")
                continue

            print(f"{label} reassembling {base_path.name} from {len(chunks)} chunk(s) ...")
            if not args.dry_run:
                try:
                    reassemble_chunks(chunks, base_path)
                except OSError as exc:
                    print(f"  reassembly failed: {exc}", file=sys.stderr)
                    failures.append((base_path, exc))
                    continue

            videos_in_archive = _count_videos_in_archive(base_path) if base_path.exists() else 0
            print(f"  {'would extract' if args.dry_run else 'extracting'} -> {out_dir} ({videos_in_archive} video(s))")
            if not args.dry_run:
                try:
                    extract_one(base_path, out_dir)
                except (tarfile.TarError, OSError) as exc:
                    print(f"  extraction failed: {exc}", file=sys.stderr)
                    failures.append((base_path, exc))
                    base_path.unlink(missing_ok=True)
                    continue
                base_path.unlink()
                for chunk in chunks:
                    chunk.unlink()
                print(f"  removed reassembled file + {len(chunks)} chunk(s)")

            total_videos_extracted += videos_in_archive

    # --- Regular archives ---
    archives = list(iter_archive_files(src, args.recursive))
    if not archives and not split_groups:
        print(f"no archives found under {src}")
        return 0

    if archives:
        print(f"found {len(archives)} regular archive(s) under {src}")

    for index, archive in enumerate(archives, start=1):
        stem = _strip_archive_suffix(archive)
        out_dir = (args.dest / stem) if args.dest else (archive.parent / stem)

        if args.skip_existing and out_dir.is_dir() and any(out_dir.iterdir()):
            print(f"[{index}/{len(archives)}] skip (exists): {archive.name}")
            continue

        videos_in_archive = _count_videos_in_archive(archive)
        action = "would extract" if args.dry_run else "extracting"
        print(f"[{index}/{len(archives)}] {action} {archive.name} -> {out_dir} ({videos_in_archive} video(s))")
        if not args.dry_run:
            try:
                if _is_zip(archive):
                    extract_zip(archive, out_dir)
                else:
                    extract_one(archive, out_dir)
            except (tarfile.TarError, zipfile.BadZipFile, OSError) as exc:
                print(f"  failed: {exc}", file=sys.stderr)
                failures.append((archive, exc))
                continue

            if args.remove_after:
                archive.unlink()
                print(f"  removed {archive.name}")

        total_videos_extracted += videos_in_archive

    if failures:
        print(f"\n{len(failures)} archive(s) failed:", file=sys.stderr)
        for archive, exc in failures:
            print(f"  {archive}: {exc}", file=sys.stderr)

    scan_root = args.dest if args.dest else src
    total_videos_now = _count_videos_in_dir(scan_root)
    print(f"\n--- statistics {'(dry run)' if args.dry_run else ''} ---")
    label = "videos that would be extracted" if args.dry_run else "videos extracted this run"
    print(f"  {label} : {total_videos_extracted}")
    print(f"  total videos under {scan_root} : {total_videos_now}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
