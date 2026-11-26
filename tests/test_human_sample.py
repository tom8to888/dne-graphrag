"""Human check on the judges: 20 blinded pairs from the test split, a rater picks per criterion."""
import pandas as pd

from dne.human_sample import make_human_sample

SYS = ["basic", "local", "global"]


def _data():
    qs = pd.DataFrame({"id": [f"Q{i:02d}" for i in range(30)],
                       "question": [f"question {i}?" for i in range(30)],
                       "kind": (["lookup", "connecting", "big_picture"] * 10),
                       "split": ["test"] * 24 + ["dev"] * 6})
    ans = pd.DataFrame([{"qid": q, "system": s, "answer": f"{s} answer to {q}"} for q in qs.id for s in SYS])
    return ans, qs


def test_twenty_blinded_items_from_the_test_split_only():
    ans, qs = _data()
    sample, key = make_human_sample(ans, qs, n=20, seed=42)
    assert len(sample) == 20 and list(sample.columns) == ["item", "qid", "question", "answer_1", "answer_2"]
    assert set(sample.qid) <= set(qs.loc[qs.split == "test", "id"])
    text = " ".join(sample.question) + " ".join(sample.columns)
    assert not any(s in text for s in SYS)  # no system names next to the answers


def test_key_maps_each_shown_answer_back_to_its_system():
    ans, qs = _data()
    sample, key = make_human_sample(ans, qs, n=20, seed=42)
    m = sample.merge(key, on="item")
    assert all(a.startswith(s) for a, s in zip(m.answer_1, m.shown_1))
    assert all(a.startswith(s) for a, s in zip(m.answer_2, m.shown_2))
    assert (m.shown_1 != m.shown_2).all()


def test_balanced_over_system_pairs_and_both_display_orders():
    ans, qs = _data()
    _, key = make_human_sample(ans, qs, n=20, seed=42)
    pairs = key.apply(lambda r: "|".join(sorted([r.shown_1, r.shown_2])), axis=1).value_counts()
    assert sorted(pairs.tolist()) == [6, 7, 7]
    assert 0 < (key.shown_1 < key.shown_2).sum() < 20  # both orders occur


def test_same_seed_same_sample():
    ans, qs = _data()
    a, _ = make_human_sample(ans, qs, n=20, seed=42)
    b, _ = make_human_sample(ans, qs, n=20, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_every_item_uses_a_different_question_when_there_are_enough():
    ans, qs = _data()  # 24 test questions
    sample, _ = make_human_sample(ans, qs, n=20, seed=42)
    assert sample.qid.nunique() == 20


def test_display_order_is_balanced_within_each_system_pair():
    ans, qs = _data()
    _, key = make_human_sample(ans, qs, n=20, seed=42)
    for pair, g in key.groupby(key.apply(lambda r: "|".join(sorted([r.shown_1, r.shown_2])), axis=1)):
        first = g.shown_1.value_counts()
        assert first.max() - (len(g) - first.max()) <= 1, pair  # e.g. 4/3 for 7 items, 3/3 for 6


# ---- ratings export + agreement (23/11) ------------------------------------------------------------------

KEY = pd.DataFrame({"item": ["H01", "H02"], "qid": ["A1", "B2"], "shown_1": ["local", "global"],
                    "shown_2": ["global", "basic"]})


def test_ratings_from_docs_maps_choices_to_systems_and_skips_missing():
    from dne.human_sample import ratings_from_docs
    docs = {"H01": {"comprehensiveness": "1", "directness": "tie", "updated": "x", "note": "hard one"},
            "H02": {"comprehensiveness": "2"}}
    r = ratings_from_docs(docs, KEY, "human1").set_index(["item", "criterion"])
    assert len(r) == 3  # unrated criteria are left out, not counted as ties
    assert r.loc[("H01", "comprehensiveness"), "winner"] == "local"
    assert r.loc[("H01", "directness"), "winner"] == "tie"
    assert r.loc[("H02", "comprehensiveness"), "winner"] == "basic"
    assert r.loc[("H01", "comprehensiveness"), "note"] == "hard one"
    assert set(r.rater) == {"human1"} and r.loc[("H02", "comprehensiveness"), "qid"] == "B2"


def test_judge_item_verdicts_nets_both_display_orders():
    from dne.human_sample import judge_item_verdicts
    j = pd.DataFrame([  # H01 = A1 local vs global; pair stored as sys_a=local, sys_b=global
        {"qid": "A1", "sys_a": "local", "sys_b": "global", "order": 0, "criterion": "diversity", "winner": "global"},
        {"qid": "A1", "sys_a": "local", "sys_b": "global", "order": 1, "criterion": "diversity", "winner": "global"},
        {"qid": "A1", "sys_a": "local", "sys_b": "global", "order": 0, "criterion": "directness", "winner": "local"},
        {"qid": "A1", "sys_a": "local", "sys_b": "global", "order": 1, "criterion": "directness", "winner": "global"},
        {"qid": "A1", "sys_a": "local", "sys_b": "basic", "order": 0, "criterion": "diversity", "winner": "basic"},
    ])
    v = judge_item_verdicts(j, KEY, "opus").set_index(["item", "criterion"])
    assert v.loc[("H01", "diversity"), "winner"] == "global" and v.loc[("H01", "diversity"), "choice"] == "2"
    assert v.loc[("H01", "directness"), "winner"] == "tie"  # split across orders
    assert len(v) == 2  # the local-vs-basic row belongs to no sample item


def test_rater_agreement_counts_and_kappa():
    from dne.human_sample import rater_agreement
    rows = []
    for item, crit, a, b in [("H01", "c1", "1", "1"), ("H01", "c2", "2", "1"), ("H02", "c1", "tie", "tie"),
                             ("H02", "c2", "2", "2")]:
        rows += [{"rater": "x", "item": item, "criterion": crit, "choice": a},
                 {"rater": "y", "item": item, "criterion": crit, "choice": b}]
    rows.append({"rater": "x", "item": "H03", "criterion": "c1", "choice": "1"})  # y did not rate H03
    t = rater_agreement(pd.DataFrame(rows)).set_index(["rater_1", "rater_2"]).loc[("x", "y")]
    assert (t["items"], t["verdicts"], t["both_decided"], t["opposite_winner"]) == (2, 4, 3, 1)
    assert t.agreement == 0.75
    # po = .75; x: 1 .25, 2 .5, tie .25; y: 1 .5, 2 .25, tie .25 → pe = .125+.125+.0625 = .3125
    assert abs(t.kappa - (0.75 - 0.3125) / (1 - 0.3125)) < 1e-9
