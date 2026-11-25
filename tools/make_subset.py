"""Write data/processed/subset_<n>.csv: first 50 usable articles + every candidate question's gold
article + a seeded random fill (dne.subset.choose_subset).

    python tools/make_subset.py 200
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dne.paths import PROCESSED, QUESTIONS  # noqa: E402
from dne.subset import choose_subset  # noqa: E402

n = int(sys.argv[1])
arts = pd.read_csv(PROCESSED / "articles.csv")
ok = arts.loc[arts.status == "ok", "aid"].astype(int).tolist()
cand = pd.read_csv(QUESTIONS / "candidates.csv", dtype=str).fillna("")
gold = {int(x) for v in cand.gold_aids for x in re.findall(r"\d+", v)}
aids = choose_subset(ok, gold, n)
out = PROCESSED / f"subset_{n}.csv"
sub = arts[arts.aid.isin(aids)][["aid", "title", "year", "n_tokens"]]
sub.to_csv(out, index=False)
print(f"{out}: {len(sub)} articles, {len(gold)} gold, {sub.n_tokens.sum():,} tokens")
