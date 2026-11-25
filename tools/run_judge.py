"""Opus judge + correctness for the main run (12k context, results/answers.parquet).

    python tools/run_judge.py          # prints the plan, no API calls
    python tools/run_judge.py --yes    # runs it (needs ANTHROPIC_API_KEY in .env)

Pairs: basic, local and global, every pair on all 60 questions in both display orders (360 calls).
Correctness: all three systems on the look-up and connecting questions (120 calls).
Resumable: pair and item ids already in the output files are skipped, so finished work is never paid twice.
Every call is logged in results/spend_ledger.csv; the run stops at the ledger's USD 20 cap.
"""
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dne.judge import build_pairs, correctness_items, run_correctness, run_judge  # noqa: E402
from dne.paths import ROOT  # noqa: E402
from dne.spend import SpendLedger  # noqa: E402

yes = "--yes" in sys.argv
load_dotenv(ROOT / ".env")
res = ROOT / "results"
answers = pd.read_parquet(res / "answers.parquet")
questions = pd.read_csv(ROOT / "data/questions/questions.csv").fillna("")
ledger = SpendLedger(res / "spend_ledger.csv")
if yes:
    import anthropic
    client = anthropic.Anthropic()
else:
    client = None

pairs = build_pairs(answers, ["basic", "local", "global"])
run_judge(pairs, answers, questions, ledger, client, res / "judgments.jsonl", yes=yes)
run_correctness(correctness_items(answers, questions), answers, questions, ledger, client,
                res / "correctness.jsonl", yes=yes)
