"""Paired significance tests on pairwise-judge results (study guide LU10: statistical tests + argued p-value)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


def holm(p: np.ndarray) -> np.ndarray:
    """Holm–Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def question_scores(judgments: pd.DataFrame) -> pd.DataFrame:
    """Per question × system pair × criterion: +1 if sys_b wins, −1 if sys_a wins, averaged over both orders.

    `judgments` columns: qid, sys_a, sys_b, order, criterion, winner (a system name or "tie"), where
    sys_a/sys_b are the two systems in a fixed canonical order (not the display order).
    """
    j = judgments.copy()
    j["score"] = np.where(j.winner == j.sys_b, 1.0, np.where(j.winner == j.sys_a, -1.0, 0.0))
    return j.groupby(["sys_a", "sys_b", "criterion", "qid"], as_index=False)["score"].mean()


def paired_tests(judgments: pd.DataFrame, questions: pd.DataFrame, alpha: float = 0.05,
                 by_kind: bool = False) -> pd.DataFrame:
    test_ids = set(questions.loc[questions.split == "test", "id"])
    s = question_scores(judgments[judgments.qid.isin(test_ids)])
    keys = ["sys_a", "sys_b", "criterion"]
    if by_kind:
        s = s.merge(questions[["id", "kind"]], left_on="qid", right_on="id").drop(columns="id")
        keys.append("kind")
    rows = []
    for key, g in s.groupby(keys):
        sc = g.score.to_numpy()
        n_b, n_a = int((sc > 0).sum()), int((sc < 0).sum())
        p_sign = binomtest(n_b, n_b + n_a, 0.5).pvalue if n_b + n_a else 1.0
        p_wil = wilcoxon(sc).pvalue if np.any(sc != 0) else 1.0
        rows.append({**dict(zip(keys, key)), "n_questions": len(sc), "n_better_b": n_b, "n_better_a": n_a,
                     "n_ties": int((sc == 0).sum()), "mean_score": float(sc.mean()),
                     "p_sign": float(p_sign), "p_wilcoxon": float(p_wil)})
    res = pd.DataFrame(rows)
    if len(res):
        res["p_holm"] = holm(res.p_sign.to_numpy())
        res["significant"] = res.p_holm < alpha
        res["p_holm_wilcoxon"] = holm(res.p_wilcoxon.to_numpy())  # the paper's test (Edge et al., App. G)
        res["significant_wilcoxon"] = res.p_holm_wilcoxon < alpha
    return res
