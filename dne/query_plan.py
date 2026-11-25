"""Pure helpers for the Kaggle query run: which (question, system) jobs to run, retrieved articles,
and the benchmark's full-run projection. Runs inside the graphrag venv (pandas only)."""
from __future__ import annotations

import pandas as pd

from dne.retrieval import text_units_to_aids


def plan_jobs(questions: pd.DataFrame, systems: list[str], done: set[tuple[str, str]],
              per_kind: int | None = None) -> list[tuple[str, str]]:
    """Question-major job list, so a stopped run has every system for each finished question.
    per_kind (benchmark): the first per_kind questions of each kind, dev questions first."""
    qs = questions
    if per_kind:
        picked = []
        for _kind, g in qs.groupby("kind", sort=False):
            g = pd.concat([g[g.split == "dev"], g[g.split != "dev"]])
            picked.append(g.head(per_kind))
        qs = pd.concat(picked)
    return [(q, s) for q in qs["id"] for s in systems if (q, s) not in done]


def sources_to_aids(context, text_units: pd.DataFrame, documents: pd.DataFrame) -> list[int]:
    """Articles behind the text units a search put in its context ('sources' records; ids are the
    text units' human_readable_id as strings, or full ids). Global search has no sources → []."""
    if not isinstance(context, dict) or not isinstance(context.get("sources"), pd.DataFrame):
        return []
    full_of = {str(h): i for h, i in zip(text_units["human_readable_id"], text_units["id"])}
    ids = [full_of.get(str(x), str(x)) for x in context["sources"]["id"]]
    return text_units_to_aids(ids, text_units, documents)


def projection_line(n_done_q: int, elapsed: float, n_total_q: int, per_system: dict[str, float]) -> str:
    """Benchmark result: projected wall time for all questions; per_system = seconds spent per system."""
    if n_done_q <= 0:
        raise ValueError("no finished question to project from")
    full_s = elapsed / n_done_q * n_total_q
    shares = ", ".join(f"{s} {t / n_done_q:.0f} s/q" for s, t in per_system.items())
    return f"projected full run ({n_total_q} q): {full_s / 60:.0f} min | {shares}"


def broken_systems(rows: list[dict], first_n: int = 3) -> list[str]:
    """Systems whose first first_n finished queries all failed: stop the run instead of burning GPU quota
    (benchmark 2: graphrag 3.3's BasicSearch.search returned empty answers for every question)."""
    by_system: dict[str, list[bool]] = {}
    for r in rows:
        by_system.setdefault(r["system"], []).append(bool(r.get("error")))
    return [s for s, errs in by_system.items() if len(errs) >= first_n and all(errs[:first_n])]
