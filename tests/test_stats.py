import numpy as np
import pandas as pd
import pytest

from dne.stats import holm, paired_tests


def _judg(n, winner_fn, crit="comprehensiveness"):
    rows = []
    for q in range(n):
        for order in (0, 1):
            rows.append({"qid": f"q{q}", "sys_a": "basic", "sys_b": "global", "order": order,
                         "criterion": crit, "winner": winner_fn(q, order)})
    return pd.DataFrame(rows)


def _qs(n, split="test"):
    return pd.DataFrame({"id": [f"q{i}" for i in range(n)], "kind": "big_picture", "split": split})


def test_paired_tests_all_wins_significant():
    res = paired_tests(_judg(20, lambda q, o: "global"), _qs(20))
    row = res.iloc[0]
    assert row.n_better_b == 20 and row.n_better_a == 0
    assert row.p_sign < 0.001 and row.p_holm < 0.001 and bool(row.significant)


def test_paired_tests_balanced_not_significant():
    res = paired_tests(_judg(20, lambda q, o: "global" if q % 2 else "basic"), _qs(20))
    assert res.iloc[0].p_sign == pytest.approx(1.0) and not bool(res.iloc[0].significant)


def test_order_disagreement_counts_as_tie():
    res = paired_tests(_judg(10, lambda q, o: "global" if o == 0 else "basic"), _qs(10))
    assert res.iloc[0].n_ties == 10


def test_only_test_split_used():
    qs = _qs(20)
    qs.loc[:9, "split"] = "dev"
    res = paired_tests(_judg(20, lambda q, o: "global"), qs)
    assert res.iloc[0].n_questions == 10


def test_holm_monotone():
    p = np.array([0.01, 0.04, 0.03, 0.2])
    adj = holm(p)
    assert adj[0] == pytest.approx(0.04)
    order = np.argsort(p)
    assert all(np.diff(adj[order]) >= 0)
    assert all(adj <= 1)


def test_paired_tests_wilcoxon_holm_column():
    # Edge et al. App. G: Wilcoxon signed-rank + Holm-Bonferroni (reported next to the sign test)
    j = pd.concat([_judg(20, lambda q, o: "global"),
                   _judg(20, lambda q, o: "global" if q % 2 else "basic", crit="diversity")])
    res = paired_tests(j, _qs(20)).set_index("criterion")
    assert np.allclose(res.p_holm_wilcoxon.to_numpy(), holm(res.p_wilcoxon.to_numpy()))
    assert res.loc["comprehensiveness", "p_holm_wilcoxon"] < 0.001
    assert bool(res.loc["comprehensiveness", "significant_wilcoxon"])
    assert not bool(res.loc["diversity", "significant_wilcoxon"])
