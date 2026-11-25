import numpy as np
import pandas as pd
import pytest

from dne.analysis import bootstrap_ci, cost_table, length_bias, win_rates


def _j(rows):
    return pd.DataFrame(rows, columns=["qid", "sys_a", "sys_b", "order", "criterion", "winner"])


QS = pd.DataFrame({"id": ["q1", "q2"], "kind": ["lookup", "big_picture"], "split": ["test", "test"]})


def test_win_rates_tie_half():
    j = _j([("q1", "basic", "global", 0, "diversity", "tie"), ("q1", "basic", "global", 1, "diversity", "tie")])
    r = win_rates(j, QS, by=["criterion"])
    assert r.loc[0, "win_rate_b"] == pytest.approx(0.5)


def test_win_rates_order_averaged():
    j = _j([("q1", "basic", "global", 0, "diversity", "global"), ("q1", "basic", "global", 1, "diversity", "basic"),
            ("q2", "basic", "global", 0, "diversity", "global"), ("q2", "basic", "global", 1, "diversity", "global")])
    r = win_rates(j, QS, by=["criterion"])
    assert r.loc[0, "win_rate_b"] == pytest.approx(0.75)


def test_win_rates_by_kind():
    j = _j([("q1", "basic", "global", 0, "diversity", "basic"), ("q1", "basic", "global", 1, "diversity", "basic"),
            ("q2", "basic", "global", 0, "diversity", "global"), ("q2", "basic", "global", 1, "diversity", "global")])
    r = win_rates(j, QS, by=["kind", "criterion"]).set_index("kind")
    assert r.loc["lookup", "win_rate_b"] == 0 and r.loc["big_picture", "win_rate_b"] == 1


def test_bootstrap_ci_contains_mean():
    df = pd.DataFrame({"qid": [f"q{i}" for i in range(50)], "x": np.linspace(0, 1, 50)})
    lo, hi = bootstrap_ci(df, lambda d: d.x.mean(), n=500, seed=1)
    assert lo < 0.5 < hi and hi - lo < 0.4


def test_cost_table_claude_conversion():
    index_run = {"wall_seconds": 3600, "gpu_hours": 2.0, "kwh": 0.1,
                 "totals": {"prompt_tokens": 1_000_000, "generation_tokens": 100_000, "requests": 10}}
    answers = pd.DataFrame({"system": ["basic"], "latency_s": [2.0], "prompt_tokens": [1000],
                            "generation_tokens": [100], "llm_calls": [1]})
    t = cost_table(index_run, answers, t4_usd_per_hour=0.35).set_index("stage")
    assert t.loc["indexing", "usd_if_sonnet_5_5"] == pytest.approx(3.00)
    assert t.loc["indexing", "usd_cloud_t4"] == pytest.approx(0.70)
    assert t.loc["answer: basic", "prompt_tokens"] == 1000


def test_length_bias_longer_wins():
    j = _j([("q1", "basic", "global", 0, "comprehensiveness", "global"),
            ("q2", "basic", "global", 0, "comprehensiveness", "basic")])
    ans = pd.DataFrame({"qid": ["q1", "q1", "q2", "q2"], "system": ["basic", "global"] * 2,
                        "answer_tokens": [100, 300, 400, 200]})
    r = length_bias(j, ans)
    assert r.share_longer_won.iloc[0] == pytest.approx(1.0)


def test_correctness_tables_shapes_and_values():
    import pandas as pd
    from dne.analysis import correctness_tables
    recs = pd.DataFrame({"qid": ["L1"] * 3 + ["C1"] * 3, "system": ["basic", "local", "global"] * 2,
                         "score": [3, 1, 1, 2, 2, 0], "coverage": ["all", "some", "some", "most", "most", "none"],
                         "contradicts": [False, True, False, False, False, True]})
    qs = pd.DataFrame({"id": ["L1", "C1"], "kind": ["lookup", "connecting"], "split": ["test", "dev"]})
    t = correctness_tables(recs, qs)
    assert set(t) == {"by_system", "coverage", "by_kind", "by_split", "paired", "per_question"}
    assert t["by_system"].loc["basic", "mean"] == 2.5 and t["by_system"].loc["global", "contradicts"] == 1
    assert t["coverage"].loc["local", "some"] == 1 and list(t["coverage"].columns) == ["all", "most", "some", "none"]
    assert t["by_kind"].loc["lookup", "basic_mean"] == 3
    assert t["per_question"].loc["C1", "global"] == 0
    row = t["paired"].set_index(["sys_1", "sys_2"]).loc[("basic", "global")]
    assert row.n_first_better == 2 and row.n_first_worse == 0


def test_judge_agreement_nets_both_display_orders():
    import pandas as pd
    from dne.analysis import judge_agreement
    # Q1: judge a wins both orders for basic on c (net: first); split on d (net: tie)
    # Q2: judge a basic once + tie (net: first) on c; local both orders on d (net: second)
    a = pd.DataFrame({"pair_id": ["Q1|basic|local|0", "Q1|basic|local|1"] * 2 + ["Q2|basic|local|0", "Q2|basic|local|1"] * 2,
                      "criterion": ["c", "c", "d", "d", "c", "c", "d", "d"],
                      "winner": ["basic", "basic", "basic", "local", "basic", "tie", "local", "local"]})
    b = a.copy()
    b["winner"] = ["basic", "local", "basic", "local", "basic", "basic", "local", "basic"]  # nets: tie, tie, first, tie
    t = judge_agreement(a, b)
    assert t.loc["all", "n"] == 4  # one netted verdict per (question, pair, criterion), not one per display order
    assert t.loc["c", "agreement"] == 0.5 and t.loc["d", "agreement"] == 0.5
    assert t.loc["all", "agreement"] == 0.5


def test_judgments_long_and_order_consistency(tmp_path):
    import json
    from dne.analysis import judgments_long, order_consistency
    p = tmp_path / "j.jsonl"
    recs = [{"pair_id": "Q1|basic|local|0", "qid": "Q1", "sys_a": "basic", "sys_b": "local", "order": 0,
             "system_winners": {"comprehensiveness": "basic", "diversity": "tie"}},
            {"pair_id": "Q1|basic|local|1", "qid": "Q1", "sys_a": "basic", "sys_b": "local", "order": 1,
             "system_winners": {"comprehensiveness": "basic", "diversity": "local"}},
            {"pair_id": "Q2|basic|local|0", "qid": "Q2", "error": "timeout"}]
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    j = judgments_long(p)
    assert len(j) == 4 and set(j.columns) >= {"pair_id", "qid", "sys_a", "sys_b", "order", "criterion", "winner"}
    oc = order_consistency(j)
    assert oc.loc["comprehensiveness", "same_both_orders"] == 1.0 and oc.loc["diversity", "same_both_orders"] == 0.0
    assert oc.loc["all", "ties"] == 0.25


def test_correctness_tables_custom_systems_and_pairs():
    import pandas as pd
    from dne.analysis import correctness_tables
    recs = pd.DataFrame({"qid": ["L1", "L1", "C1", "C1"], "system": ["basic_8k", "global_8k"] * 2,
                         "score": [3, 1, 2, 2], "coverage": ["all", "some", "most", "most"],
                         "contradicts": [False, True, False, False]})
    qs = pd.DataFrame({"id": ["L1", "C1"], "kind": ["lookup", "connecting"], "split": ["test", "test"]})
    t = correctness_tables(recs, qs, systems=["basic_8k", "global_8k"], pairs=[("basic_8k", "global_8k")])
    assert list(t["by_system"].index) == ["basic_8k", "global_8k"]
    assert t["by_system"].loc["basic_8k", "mean"] == 2.5
    assert len(t["paired"]) == 1
    row = t["paired"].iloc[0]
    assert (row.sys_1, row.sys_2, row.n_first_better, row.n_equal) == ("basic_8k", "global_8k", 1, 1)
