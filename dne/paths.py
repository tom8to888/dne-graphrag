"""Project paths. Works from any cwd inside the repo (e.g. notebooks/)."""
from __future__ import annotations

from pathlib import Path


def find_root(start: Path | None = None) -> Path:
    """Walk up from `start` (default: this file) to the dir holding pyproject.toml."""
    here = Path(start or __file__).resolve()
    for d in [here, *here.parents]:
        if (d / "pyproject.toml").is_file():
            return d
    raise FileNotFoundError(f"no pyproject.toml above {here}")


ROOT = find_root()
RAW = ROOT / "data/raw/itf_ccsr"
PDF_DIR = RAW / "pdf"
META_DIR = RAW / "meta"
PROCESSED = ROOT / "data/processed"
TEXT_DIR = PROCESSED / "text"
QUESTIONS = ROOT / "data/questions"
RESULTS = ROOT / "results"
FIGURES = ROOT / "report/figures"
