"""Blinded human check of the judges.

20 (question, system pair) items from the TEST split, spread over the three system pairs (7/7/6) and, within
each pair, round-robin over question kinds; each item shows the two answers in an order that alternates within each pair (balanced) with no system
names. The key (which system is answer 1 / answer 2) is kept apart and only used afterwards to score
agreement with the Opus and Mistral judges.

    python -m dne.human_sample      → results/human_sample.csv, results/human_sample_key.csv
"""
from __future__ import annotations

import random

import pandas as pd

PAIRS = [("basic", "local"), ("basic", "global"), ("local", "global")]


def make_human_sample(answers: pd.DataFrame, questions: pd.DataFrame, n: int = 20,
                      seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    test = questions[questions.split == "test"]
    by_kind = {k: sorted(g.id) for k, g in test.groupby("kind")}
    for ids in by_kind.values():
        rng.shuffle(ids)
    kinds = sorted(by_kind)
    per_pair = [n // len(PAIRS) + (1 if i < n % len(PAIRS) else 0) for i in range(len(PAIRS))]
    ans = {(r.qid, r.system): r.answer for r in answers.itertuples()}
    qtext = dict(zip(questions.id, questions.question))
    used, items = set(), []
    for (a, b), k in zip(PAIRS, per_pair):
        picked, turn, flip = 0, 0, rng.random() < 0.5
        while picked < k:
            pool = by_kind[kinds[turn % len(kinds)]]
            turn += 1
            taken = {i["qid"] for i in items}
            qid = next((q for q in pool if q not in taken), None)  # a new question for every item
            if qid is None and turn > 3 * len(kinds):  # fewer test questions than items: allow repeats
                qid = next((q for q in pool if (q, a, b) not in used), None)
            if qid is None:
                if turn > 6 * len(kinds):
                    raise ValueError("not enough test questions for the sample")
                continue
            used.add((qid, a, b))
            shown = (a, b) if flip else (b, a)  # alternate: each system shown first equally often
            flip = not flip
            items.append({"qid": qid, "shown_1": shown[0], "shown_2": shown[1]})
            picked += 1
    rng.shuffle(items)
    sample = pd.DataFrame([{"item": f"H{i + 1:02d}", "qid": it["qid"], "question": qtext[it["qid"]],
                            "answer_1": ans[(it["qid"], it["shown_1"])], "answer_2": ans[(it["qid"], it["shown_2"])]}
                           for i, it in enumerate(items)])
    key = pd.DataFrame([{"item": f"H{i + 1:02d}", "qid": it["qid"], "shown_1": it["shown_1"],
                         "shown_2": it["shown_2"]} for i, it in enumerate(items)])
    return sample, key


if __name__ == "__main__":
    from dne.paths import ROOT
    s, k = make_human_sample(pd.read_parquet(ROOT / "results/answers.parquet"),
                             pd.read_csv(ROOT / "data/questions/questions.csv", dtype=str, keep_default_na=False))
    s.to_csv(ROOT / "results/human_sample.csv", index=False)
    k.to_csv(ROOT / "results/human_sample_key.csv", index=False)
    print(f"{len(s)} items → results/human_sample.csv (+ key)")


# ---- ratings export + agreement ------------------------------------------------------------------------

CRITERIA = ("comprehensiveness", "diversity", "empowerment", "directness")


def ratings_from_docs(docs: dict[str, dict], key: pd.DataFrame, rater: str) -> pd.DataFrame:
    """The rating page's saved documents ({item: {criterion: "1" | "2" | "tie", "note": ...}}) → one row per rated
    (item, criterion): rater, item, qid, criterion, choice, winner (the system behind the chosen answer, or "tie"),
    note. Criteria a rater has not rated yet are left out."""
    k = key.set_index("item")
    rows = []
    for item, d in sorted(docs.items()):
        if item not in k.index:
            continue
        s1, s2 = k.at[item, "shown_1"], k.at[item, "shown_2"]
        for c in CRITERIA:
            if c in d:
                ch = str(d[c])
                rows.append({"rater": rater, "item": item, "qid": k.at[item, "qid"], "criterion": c, "choice": ch,
                             "winner": {"1": s1, "2": s2}.get(ch, "tie"), "note": d.get("note", "")})
    return pd.DataFrame(rows, columns=["rater", "item", "qid", "criterion", "choice", "winner", "note"])


def judge_item_verdicts(judgments: pd.DataFrame, key: pd.DataFrame, rater: str) -> pd.DataFrame:
    """An LLM judge's long verdicts (both display orders) → one verdict per sample item and criterion, in the same
    shape as ratings_from_docs: the system that won more of the two orders, "tie" when the orders split or tie."""
    rows = []
    for k in key.itertuples():
        pair = {k.shown_1, k.shown_2}
        g = judgments[(judgments.qid == k.qid) & judgments.sys_a.isin(pair) & judgments.sys_b.isin(pair)]
        for c, gc in g.groupby("criterion"):
            net = int((gc.winner == k.shown_1).sum() - (gc.winner == k.shown_2).sum())
            ch = "1" if net > 0 else "2" if net < 0 else "tie"
            rows.append({"rater": rater, "item": k.item, "qid": k.qid, "criterion": c, "choice": ch,
                         "winner": {"1": k.shown_1, "2": k.shown_2}.get(ch, "tie"), "note": ""})
    return pd.DataFrame(rows, columns=["rater", "item", "qid", "criterion", "choice", "winner", "note"])


def rater_agreement(verdicts: pd.DataFrame) -> pd.DataFrame:
    """Every pair of raters (humans and judges) on the (item, criterion) both rated: raw agreement, Cohen's kappa
    on the choice (first answer / second answer / tie), how often both picked a winner and how often opposite ones."""
    from itertools import combinations

    from dne.analysis import cohen_kappa
    wide = verdicts.pivot_table(index=["item", "criterion"], columns="rater", values="choice", aggfunc="first")
    rows = []
    for a, b in combinations(list(dict.fromkeys(verdicts.rater)), 2):
        both = wide[[a, b]].dropna()
        x, y = both[a], both[b]
        dec = (x != "tie") & (y != "tie")
        rows.append({"rater_1": a, "rater_2": b, "items": both.index.get_level_values(0).nunique(),
                     "verdicts": len(both), "agreement": float((x == y).mean()),
                     "kappa": cohen_kappa(x, y) if len(both) else float("nan"),
                     "both_decided": int(dec.sum()), "opposite_winner": int((dec & (x != y)).sum())})
    return pd.DataFrame(rows)
