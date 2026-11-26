"""Pick the article subset for a size-limited index (200 of 541 articles, to fit Kaggle's 12 h session limit)."""
from __future__ import annotations

import numpy as np


def choose_subset(ok_aids: list[int], must_include: set[int], n: int, nested_first: int = 50,
                  seed: int = 42) -> list[int]:
    """First `nested_first` usable articles (the pilot/prompt-run set) + every question's gold article,
    filled at random (fixed seed) to n. Sorted aids."""
    ok = sorted(ok_aids)
    unknown = sorted(set(must_include) - set(ok))
    if unknown:
        raise ValueError(f"gold articles not in the usable corpus: {unknown}")
    base = set(ok[:nested_first]) | set(must_include)
    if len(base) > n:
        raise ValueError(f"n={n} is smaller than the {len(base)} articles that must be included")
    rest = [a for a in ok if a not in base]
    fill = np.random.default_rng(seed).choice(rest, n - len(base), replace=False).tolist()
    return sorted(base | {int(a) for a in fill})
