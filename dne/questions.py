"""The coach-style question set: loading, validation and the dev/test split."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KINDS = ("lookup", "connecting", "big_picture")
TOPICS = ("strategy", "serve_return", "mental", "fitness_heat", "arm_equipment", "other")
SPLITS = ("dev", "test", "")


def _aids(cell) -> list[int]:
    if pd.isna(cell) or str(cell).strip() == "":
        return []
    return [int(float(x)) for x in str(cell).split(";") if x.strip()]


def load_questions(path: Path, known_aids: set[int]) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["gold_aids"] = df["gold_aids"].map(_aids)
    df["split"] = df["split"].str.strip()
    dup = df.loc[df.id.duplicated(), "id"].tolist()
    if dup:
        raise ValueError(f"duplicate ids: {dup}")
    for r in df.itertuples():
        problems = []
        if r.kind not in KINDS:
            problems.append(f"kind {r.kind!r}")
        if r.topic not in TOPICS:
            problems.append(f"topic {r.topic!r}")
        if r.split not in SPLITS:
            problems.append(f"split {r.split!r}")
        if not r.question.strip():
            problems.append("empty question")
        unknown = [a for a in r.gold_aids if a not in known_aids]
        if unknown:
            problems.append(f"gold aids not in corpus {unknown}")
        if r.kind == "lookup" and not r.gold_aids:
            problems.append("lookup without gold aid")
        if problems:
            raise ValueError(f"question {r.id}: " + "; ".join(problems))
    return df.reset_index(drop=True)


def assign_split(df: pd.DataFrame, n_dev_per_kind: int = 5, seed: int = 42) -> pd.DataFrame:
    """Fill empty `split` cells: per kind, dev until the kind has n_dev_per_kind dev questions, rest test."""
    df = df.copy()
    rng = np.random.default_rng(seed)
    for kind, grp in df.groupby("kind", sort=True):
        need = max(0, n_dev_per_kind - int((grp.split == "dev").sum()))
        empty = grp.index[grp.split == ""].to_numpy()
        rng_order = rng.permutation(empty)
        df.loc[rng_order[:need], "split"] = "dev"
        df.loc[rng_order[need:], "split"] = "test"
    return df


def select_kept(candidates: pd.DataFrame) -> pd.DataFrame:
    """Rows the author marked in the `keep` column (x / y / yes / 1, any case)."""
    mark = candidates["keep"].fillna("").astype(str).str.strip().str.lower()
    return candidates[mark.isin({"x", "y", "yes", "1"})].reset_index(drop=True)
