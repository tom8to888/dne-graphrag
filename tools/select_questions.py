"""Build data/questions/questions.csv from the rows the author marked in candidates.csv (keep = x).

Empty `split` cells get a stratified dev/test split (5 dev questions per kind)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from dne.paths import PROCESSED, QUESTIONS  # noqa: E402
from dne.questions import assign_split, load_questions, select_kept  # noqa: E402

arts = pd.read_csv(PROCESSED / "articles.csv")
known = set(arts.loc[arts.status == "ok", "aid"])
cands = load_questions(QUESTIONS / "candidates.csv", known)
kept = assign_split(select_kept(cands).drop(columns="keep"))
kept["gold_aids"] = kept.gold_aids.map(lambda l: ";".join(map(str, l)))
kept.to_csv(QUESTIONS / "questions.csv", index=False)
print(kept.groupby(["kind", "split"]).size().unstack(fill_value=0))
