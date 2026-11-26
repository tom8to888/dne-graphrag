"""Judge + correctness for the 8k-context run (320 calls).

    python tools/run_judge_8k.py          # prints the plan, no API calls
    python tools/run_judge_8k.py --yes    # runs it (needs ANTHROPIC_API_KEY in .env)

Pairs (60 questions × 2 orders each): basic_8k vs global_8k (the paper's equal-context comparison) and
global (12k) vs global_8k (does the context budget matter for global search?).
Correctness: both 8k systems on the look-up + connecting questions (80 answers).
New output files; the 12k judgments/correctness files are not touched. Same ledger, same $20 cap.
"""
import sys
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dne.judge import build_pairs, correctness_items, run_correctness, run_judge  # noqa: E402
from dne.paths import ROOT  # noqa: E402
from dne.spend import SpendLedger  # noqa: E402

yes = "--yes" in sys.argv
load_dotenv(ROOT / ".env")
res = ROOT / "results"
a8 = pd.read_parquet(res / "answers_8k.parquet")
a12 = pd.read_parquet(res / "answers.parquet")
answers = pd.concat([a8, a12[a12.system == "global"]], ignore_index=True)
questions = pd.read_csv(ROOT / "data/questions/questions.csv").fillna("")
ledger = SpendLedger(res / "spend_ledger.csv")
client = anthropic.Anthropic() if yes else None

pairs = build_pairs(a8, ["basic_8k", "global_8k"]) + build_pairs(a8, ["global", "global_8k"])
run_judge(pairs, answers, questions, ledger, client, res / "judgments_8k.jsonl", yes=yes,
          errors=res / "judge_errors_8k.csv")
run_correctness(correctness_items(a8, questions), a8, questions, ledger, client, res / "correctness_8k.jsonl",
                yes=yes, errors=res / "correctness_errors_8k.csv")
