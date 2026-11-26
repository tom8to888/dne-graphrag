"""Human sample ratings → results/human_ratings.csv (one row per rater, item and rated criterion).

    python tools/export_human_ratings.py human1=<dir> human2=<dir>

Each <dir> holds one JSON file per item (H01.json ...) as saved from the rating page's database
(owner: collection `ratings`; second rater: `raters/<id>/items`). Re-run whenever more items are rated;
the CSV is rewritten. Rater ids from the page are not stored, only the names given here.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dne.human_sample import ratings_from_docs  # noqa: E402
from dne.paths import ROOT  # noqa: E402

key = pd.read_csv(ROOT / "results/human_sample_key.csv")
frames = []
for arg in sys.argv[1:]:
    rater, d = arg.split("=", 1)
    docs = {p.stem: json.loads(p.read_text()) for p in sorted(Path(d).glob("H*.json"))}
    frames.append(ratings_from_docs(docs, key, rater))
out = pd.concat(frames, ignore_index=True)
out.to_csv(ROOT / "results/human_ratings.csv", index=False)
print(out.groupby("rater").agg(items=("item", "nunique"), verdicts=("item", "size")).to_string())
