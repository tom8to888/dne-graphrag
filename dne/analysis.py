"""Win rates, bootstrap CIs, cost tables and the length-bias check for the report.

Judgments use the canonical form of dne.stats: qid, sys_a, sys_b, order, criterion, winner (system name or "tie").
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from dne.spend import cost_usd
from dne.stats import question_scores


def win_rates(judgments: pd.DataFrame, questions: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Share of wins for sys_b (tie = half a win), averaged over both display orders, then over questions."""
    s = question_scores(judgments)  # +1 b wins, -1 a wins, averaged over orders
    s["win_b"] = (s.score + 1) / 2
    s = s.merge(questions[["id", "kind"]], left_on="qid", right_on="id", how="left").drop(columns="id")
    keys = ["sys_a", "sys_b"] + by
    return (s.groupby(keys, as_index=False)
             .agg(win_rate_b=("win_b", "mean"), n_questions=("qid", "nunique")))


def bootstrap_ci(df: pd.DataFrame, stat_fn: Callable[[pd.DataFrame], float], group_col: str = "qid",
                 n: int = 2000, seed: int = 42, level: float = 0.95) -> tuple[float, float]:
    """Percentile CI, resampling whole questions (not individual judgments)."""
    rng = np.random.default_rng(seed)
    groups = {g: d for g, d in df.groupby(group_col)}
    keys = list(groups)
    stats = [stat_fn(pd.concat([groups[k] for k in rng.choice(keys, len(keys))])) for _ in range(n)]
    a = (1 - level) / 2
    return float(np.quantile(stats, a)), float(np.quantile(stats, 1 - a))


def cost_table(index_run: dict, answers: pd.DataFrame, t4_usd_per_hour: float) -> pd.DataFrame:
    """Indexing cost and mean per-answer cost per system; $ at a cloud T4 rate and as if run on Claude."""
    tot = index_run["totals"]
    rows = [{"stage": "indexing", "wall_s": index_run["wall_seconds"], "gpu_hours": index_run["gpu_hours"],
             "kwh": index_run["kwh"], "prompt_tokens": tot["prompt_tokens"],
             "generation_tokens": tot["generation_tokens"], "llm_calls": tot["requests"]}]
    for system, g in answers.groupby("system"):
        rows.append({"stage": f"answer: {system}", "wall_s": g.latency_s.mean(),
                     "gpu_hours": g.latency_s.mean() * index_run["gpu_hours"] / index_run["wall_seconds"],
                     "kwh": np.nan, "prompt_tokens": g.prompt_tokens.mean(),
                     "generation_tokens": g.generation_tokens.mean(), "llm_calls": g.llm_calls.mean()})
    t = pd.DataFrame(rows)
    t["usd_cloud_t4"] = t.gpu_hours * t4_usd_per_hour
    for model, col in (("claude-haiku-5-5", "usd_if_haiku_5_5"), ("claude-sonnet-5-5", "usd_if_sonnet_5_5")):
        t[col] = [cost_usd(model, p, g) for p, g in zip(t.prompt_tokens, t.generation_tokens)]
    return t


def length_bias(judgments: pd.DataFrame, answers: pd.DataFrame) -> pd.DataFrame:
    """Per criterion: among decided judgments, how often the longer answer won."""
    length = {(r.qid, r.system): r.answer_tokens for r in answers.itertuples()}
    j = judgments[judgments.winner != "tie"].copy()
    j["longer"] = [a if length[(q, a)] > length[(q, b)] else b for q, a, b in zip(j.qid, j.sys_a, j.sys_b)]
    j["longer_won"] = j.longer == j.winner
    return j.groupby("criterion", as_index=False).agg(share_longer_won=("longer_won", "mean"),
                                                      n=("longer_won", "size"))


CORRECT_SYSTEMS = ["basic", "local", "global"]


def correctness_tables(records: pd.DataFrame, questions: pd.DataFrame, systems: list[str] | None = None,
                       pairs: list[tuple[str, str]] | None = None) -> dict[str, pd.DataFrame]:
    """The correctness check as report tables (21/11): per system, coverage counts, per kind and split (mean +
    contradictions), paired per-question comparison (Wilcoxon, Holm over the pairs) and the per-question grid.
    systems/pairs default to the 12k run's three systems; the 8k run passes its own (23/11)."""
    from scipy.stats import wilcoxon

    from dne.stats import holm
    r = records.merge(questions[["id", "kind", "split"]], left_on="qid", right_on="id", how="left")
    s = list(systems or CORRECT_SYSTEMS)
    by_system = (r.groupby("system").agg(n=("score", "size"), mean=("score", "mean"), sd=("score", "std"),
                                         contradicts=("contradicts", "sum")).reindex(s))
    coverage = pd.crosstab(r.system, r.coverage).reindex(index=s, columns=["all", "most", "some", "none"],
                                                         fill_value=0)

    def by(col):
        m = r.pivot_table(index=col, columns="system", values="score", aggfunc="mean").reindex(columns=s)
        c = r.pivot_table(index=col, columns="system", values="contradicts", aggfunc="sum").reindex(columns=s)
        n = r.groupby(col).qid.nunique().rename("n_questions")
        return pd.concat([n, m.add_suffix("_mean"), c.add_suffix("_contradicts")], axis=1)

    grid = r.pivot_table(index="qid", columns="system", values="score").reindex(columns=s)
    rows = []
    for a, b in pairs or [("basic", "local"), ("basic", "global"), ("local", "global")]:
        d = grid[a] - grid[b]
        p = wilcoxon(grid[a], grid[b]).pvalue if (d != 0).any() else 1.0
        rows.append({"sys_1": a, "sys_2": b, "n_questions": len(d), "n_first_better": int((d > 0).sum()),
                     "n_first_worse": int((d < 0).sum()), "n_equal": int((d == 0).sum()),
                     "mean_diff": float(d.mean()), "p_wilcoxon": float(p)})
    paired = pd.DataFrame(rows)
    paired["p_holm"] = holm(paired.p_wilcoxon.to_numpy())
    return {"by_system": by_system, "coverage": coverage, "by_kind": by("kind"), "by_split": by("split"),
            "paired": paired, "per_question": grid}


def cohen_kappa(x: pd.Series, y: pd.Series) -> float:
    """Cohen's kappa of two aligned label series."""
    po = float((x.values == y.values).mean())
    px, py = x.value_counts(normalize=True), y.value_counts(normalize=True)
    pe = float(sum(px.get(k, 0) * py.get(k, 0) for k in set(px.index) | set(py.index)))
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def _netted(j: pd.DataFrame) -> pd.DataFrame:
    """Both display orders of a (question, system pair, criterion) netted into one verdict, as for the human
    sample: 'first' / 'second' when the pair's first or second system won more of the two orders, else 'tie'.
    pair_id is "qid|sys_1|sys_2|order"."""
    p = j["pair_id"].str.split("|", expand=True)
    d = j.assign(item=p[0] + "|" + p[1] + "|" + p[2],
                 v=(j["winner"] == p[1]).astype(int) - (j["winner"] == p[2]).astype(int))
    s = d.groupby(["item", "criterion"], as_index=False)["v"].sum()
    s["winner"] = s["v"].map(lambda v: "first" if v > 0 else ("second" if v < 0 else "tie"))
    return s[["item", "criterion", "winner"]]


def judge_agreement(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Two judges: raw agreement and Cohen's kappa per criterion and overall, on one netted verdict per
    (question, system pair, criterion) (see _netted). Inputs have columns pair_id, criterion, winner
    (a system name or "tie")."""
    m = _netted(a).merge(_netted(b), on=["item", "criterion"], suffixes=("_a", "_b"))
    rows = {}
    for key, g in list(m.groupby("criterion")) + [("all", m)]:
        same = g.winner_a == g.winner_b
        kappa = 1.0 if same.all() else cohen_kappa(g.winner_a, g.winner_b)
        rows[key] = {"n": len(g), "agreement": float(same.mean()), "kappa": kappa}
    return pd.DataFrame.from_dict(rows, orient="index")


def judgments_long(path) -> pd.DataFrame:
    """A judge's jsonl (Opus or the local judge) → one row per (pair, criterion); error rows are skipped."""
    import json
    from pathlib import Path
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error"):
            continue
        for c, w in r["system_winners"].items():
            rows.append({"pair_id": r["pair_id"], "qid": r["qid"], "sys_a": r["sys_a"], "sys_b": r["sys_b"],
                         "order": r["order"], "criterion": c, "winner": w})
    return pd.DataFrame(rows)


def order_consistency(judgments: pd.DataFrame) -> pd.DataFrame:
    """Per criterion and overall: share of (question, pair) with the same verdict in both display orders, and
    the share of tie verdicts."""
    piv = judgments.pivot_table(index=["qid", "sys_a", "sys_b", "criterion"], columns="order", values="winner",
                                aggfunc="first").dropna().reset_index()
    piv["same"] = piv[0] == piv[1]
    out = {c: {"same_both_orders": float(g.same.mean()),
               "ties": float((judgments[judgments.criterion == c].winner == "tie").mean())}
           for c, g in piv.groupby("criterion")}
    out["all"] = {"same_both_orders": float(piv.same.mean()), "ties": float((judgments.winner == "tie").mean())}
    return pd.DataFrame.from_dict(out, orient="index")
