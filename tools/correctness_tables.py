"""Write the correctness-check tables as CSV.

    python tools/correctness_tables.py

results/correctness_tables/*.csv        main result: test split only (same split as the judge tests)
results/correctness_tables/all40/*.csv  robustness: all 40 look-up + connecting questions (dev + test)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from dne.analysis import correctness_tables  # noqa: E402
from dne.paths import ROOT  # noqa: E402

recs = pd.DataFrame([json.loads(x) for x in (ROOT / "results/correctness.jsonl").read_text().splitlines() if x])
qs = pd.read_csv(ROOT / "data/questions/questions.csv", dtype=str, keep_default_na=False)
test_ids = set(qs.loc[qs.split == "test", "id"])
base = ROOT / "results/correctness_tables"
for out, r in ((base, recs[recs.qid.isin(test_ids)]), (base / "all40", recs)):
    out.mkdir(parents=True, exist_ok=True)
    for name, t in correctness_tables(r, qs).items():
        if out == base and name == "by_split":
            continue  # test only: a split table would have one row
        t.round(4).to_csv(out / f"{name}.csv", index=name != "paired")
        print(f"{out.relative_to(ROOT)}/{name}.csv  {t.shape}")
