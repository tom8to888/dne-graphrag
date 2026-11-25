"""Split a long graphrag index over several Kaggle sessions (12 h limit per session).

graphrag caches every LLM response under <root>/cache, keyed by the request. Session 1 stops
cleanly before the limit and its cache is packed into one archive (one file: the Kaggle CLI
uploads files one by one); session 2 restores it and reruns the same index — finished calls come
from the cache, the rest run. Same settings, prompts and model are required for cache hits.
"""
from __future__ import annotations

import os
import shutil
import tarfile
from pathlib import Path

ARCHIVE = "cache.tar.gz"


def pack_cache(graphrag_root: Path, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        tar.add(Path(graphrag_root) / "cache", arcname="cache")
    return out


def restore_cache(archive: Path, graphrag_root: Path) -> int:
    """Unpack (or copy, if Kaggle already unpacked it) into <root>/cache; returns the number of
    cached extraction calls restored."""
    root = Path(graphrag_root)
    root.mkdir(parents=True, exist_ok=True)
    if Path(archive).is_dir():
        shutil.copytree(archive, root / "cache", dirs_exist_ok=True)
        ex = root / "cache" / "extract_graph"
        return sum(1 for _ in ex.iterdir()) if ex.exists() else 0
    with tarfile.open(archive, "r:gz") as tar:
        try:
            tar.extractall(root, filter="data")
        except TypeError:  # Python without extraction filters
            tar.extractall(root)
    ex = root / "cache" / "extract_graph"
    return sum(1 for _ in ex.iterdir()) if ex.exists() else 0


def find_cache_archive(input_root: Path) -> Path | None:
    """cache.tar.gz, or the cache/ folder Kaggle makes by unpacking it on upload."""
    for dirpath, dirs, files in os.walk(input_root):
        if ARCHIVE in files:
            return Path(dirpath) / ARCHIVE
        if Path(dirpath).name == "cache" and "extract_graph" in dirs:
            return Path(dirpath)
    return None


def session_time_up(t0: float, now: float, stop_after_hours: float) -> bool:
    return now - t0 >= stop_after_hours * 3600


OUTPUT_ARCHIVE = "output.tar.gz"


def pack_output(graphrag_root: Path, out: Path) -> Path:
    """The finished tables of a stopped session, so the next one can run only the missing workflows."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        tar.add(Path(graphrag_root) / "output", arcname="output")
    return out


def restore_output(src: Path, graphrag_root: Path) -> int:
    """Unpack (or copy, if Kaggle already unpacked it) into <root>/output; returns the number of tables."""
    root = Path(graphrag_root)
    root.mkdir(parents=True, exist_ok=True)
    if Path(src).is_dir():
        shutil.copytree(src, root / "output", dirs_exist_ok=True)
    else:
        with tarfile.open(src, "r:gz") as tar:
            try:
                tar.extractall(root, filter="data")
            except TypeError:
                tar.extractall(root)
    return sum(1 for _ in (root / "output").glob("*.parquet"))


def find_output(input_root: Path) -> Path | None:
    """output.tar.gz, or the output/ folder Kaggle makes by unpacking it on upload."""
    for dirpath, dirs, files in os.walk(input_root):
        if OUTPUT_ARCHIVE in files:
            return Path(dirpath) / OUTPUT_ARCHIVE
        if Path(dirpath).name == "output" and "entities.parquet" in files:
            return Path(dirpath)
    return None
