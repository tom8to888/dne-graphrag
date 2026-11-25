"""LLM judges with Claude Opus 5.5, under the spend ledger.

1. Pairwise judge on the GraphRAG paper's four criteria; never sees the author's expected-answer notes, so the
   results stay comparable with the paper.
2. Correctness check: one call per answer, graded against the author's note of what a good answer
   covers. Only look-up and connecting questions have such notes; big-picture notes list supporting
   articles, so those are left to the human sample.

Rules: nothing is called without yes=True (--yes), so every paid run is started on purpose;
hard cap $20 via SpendLedger.check; resumable (finished pair_ids are skipped, so nothing is paid twice).
"""
from __future__ import annotations

import csv
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dne.spend import BudgetExceeded, SpendLedger

MODEL = "claude-opus-5-5"
CRITERIA = ("comprehensiveness", "diversity", "empowerment", "directness")
CRITERIA_TEXT = {
    "comprehensiveness": "How much detail does the answer provide to cover all aspects and details of the question?",
    "diversity": "How varied and rich is the answer in providing different perspectives and insights on the question?",
    "empowerment": "How well does the answer help the reader understand and make informed judgements about the topic?",
    "directness": "How specifically and clearly does the answer address the question?",
}
SYSTEM = ("You are an impartial evaluator of answers written for a tennis coach who advises adult club players. "
          "Judge only the two answers shown, against each criterion separately. Do not let the order of the "
          "answers or their length alone decide; a shorter answer can win.")
SCHEMA = {
    "type": "object",
    "properties": {c: {"type": "object",
                       "properties": {"winner": {"type": "string", "enum": ["A", "B", "tie"]},
                                      "reason": {"type": "string"}},
                       "required": ["winner", "reason"], "additionalProperties": False} for c in CRITERIA},
    "required": list(CRITERIA),
    "additionalProperties": False,
}
FIRST_CALL_ESTIMATE_USD = 0.05


@dataclass(frozen=True)
class Pair:
    pair_id: str
    qid: str
    sys_a: str
    sys_b: str
    order: int  # 0: answer A = sys_a; 1: answer A = sys_b


def build_pairs(answers: pd.DataFrame, systems: list[str]) -> list[Pair]:
    pairs = []
    for qid in answers["qid"].drop_duplicates():
        for a, b in itertools.combinations(systems, 2):
            for order in (0, 1):
                pairs.append(Pair(f"{qid}|{a}|{b}|{order}", qid, a, b, order))
    return pairs


def to_system_winners(pair: Pair, criterion_winners: dict[str, str]) -> dict[str, str]:
    shown_a, shown_b = (pair.sys_a, pair.sys_b) if pair.order == 0 else (pair.sys_b, pair.sys_a)
    return {c: {"A": shown_a, "B": shown_b}.get(w, "tie") for c, w in criterion_winners.items()}


def judge_prompt(question: str, answer_a: str, answer_b: str) -> str:
    crit = "\n".join(f"- {c}: {t}" for c, t in CRITERIA_TEXT.items())
    return (f"Question:\n{question}\n\n---\nAnswer A:\n{answer_a}\n\n---\nAnswer B:\n{answer_b}\n\n---\n"
            f"For each criterion, say which answer is better (A, B, or tie) and give a one-sentence reason.\n{crit}")


def judge_one(client, pair: Pair, question: str, answer_a: str, answer_b: str, ledger: SpendLedger,
              model: str = MODEL) -> dict:
    resp = client.beta.messages.create(
        model=model, max_tokens=16000, system=SYSTEM,
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": judge_prompt(question, answer_a, answer_b)}],
    )
    ledger.record("judge", resp.model, resp.usage.input_tokens, resp.usage.output_tokens)
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"refusal on {pair.pair_id}")
    data = json.loads(next(b.text for b in resp.content if b.type == "text"))
    winners = {c: data[c]["winner"] for c in CRITERIA}
    return {"pair_id": pair.pair_id, "qid": pair.qid, "sys_a": pair.sys_a, "sys_b": pair.sys_b, "order": pair.order,
            "criterion_winners": winners, "system_winners": to_system_winners(pair, winners),
            "reasons": {c: data[c]["reason"] for c in CRITERIA}, "model_used": resp.model}


def _done(out: Path) -> set[str]:
    if not out.exists():
        return set()
    return {json.loads(line)["pair_id"] for line in out.read_text().splitlines() if line.strip()}


def _paid_loop(todo: list, out: Path, ledger: SpendLedger, errors: Path, *, yes: bool, label: str,
               first_estimate: float) -> None:
    """Shared loop: print the plan, stop unless yes, check the cap before every call, log failures, append results."""
    print(f"{label} plan: {len(todo)} calls on {MODEL}, ≈${len(todo) * first_estimate:.2f} at the first-call "
          f"estimate, ledger so far ${ledger.total():.2f} (cap ${ledger.cap_usd:.0f})")
    if not yes:
        print("No API calls made. Re-run with --yes to run it.")
        return
    costs: list[float] = []
    for key, fn in todo:
        estimate = sum(costs) / len(costs) if costs else first_estimate
        try:
            ledger.check(estimate)
        except BudgetExceeded as e:
            print(f"stopped at the cap: {e}")
            return
        before = ledger.total()
        try:
            rec = fn()
        except Exception as e:  # noqa: BLE001 — any failure: log it, keep going, never retry silently
            with errors.open("a", newline="") as f:
                csv.writer(f).writerow([key, type(e).__name__, str(e)[:200]])
            continue
        finally:
            spent = ledger.total() - before
            if spent > 0:
                costs.append(spent)
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    print(f"{label} done: ledger ${ledger.total():.2f}")


def run_judge(pairs: list[Pair], answers: pd.DataFrame, questions: pd.DataFrame, ledger: SpendLedger, client,
              out: Path, *, yes: bool = False, limit: int | None = None, errors: Path | None = None) -> None:
    out, errors = Path(out), Path(errors or Path(out).with_name("judge_errors.csv"))
    ans = {(r.qid, r.system): r.answer for r in answers.itertuples()}
    qtext = dict(zip(questions["id"], questions["question"]))

    def job(p: Pair):
        shown = (p.sys_a, p.sys_b) if p.order == 0 else (p.sys_b, p.sys_a)
        return lambda: judge_one(client, p, qtext[p.qid], ans[(p.qid, shown[0])], ans[(p.qid, shown[1])], ledger)

    todo = [(p.pair_id, job(p)) for p in pairs if p.pair_id not in _done(out)][:limit]
    _paid_loop(todo, out, ledger, errors, yes=yes, label="judge", first_estimate=FIRST_CALL_ESTIMATE_USD)


# ---- correctness check against the author's expected-answer notes -------------------------------------------

COVERAGE_SCORE = {"all": 3, "most": 2, "some": 1, "none": 0}
CORRECT_SYSTEM = ("You check answers written for a tennis coach who advises adult club players. You get a question, "
                  "the coach's short note of the key points a good answer contains (taken from the source article), "
                  "and one answer. Grade only whether the answer conveys those key points and whether it says "
                  "anything that contradicts them. Extra material that does not contradict the note is neither "
                  "rewarded nor penalised; length and style do not count.")
CORRECT_SCHEMA = {
    "type": "object",
    "properties": {
        "coverage": {"type": "string", "enum": list(COVERAGE_SCORE)},
        "contradicts": {"type": "boolean"},
        "missing": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["coverage", "contradicts", "missing", "reason"],
    "additionalProperties": False,
}
CORRECT_FIRST_CALL_USD = 0.02
CORRECT_KINDS = ("lookup", "connecting")


@dataclass(frozen=True)
class Item:
    item_id: str
    qid: str
    system: str


def correctness_items(answers: pd.DataFrame, questions: pd.DataFrame) -> list[Item]:
    """One item per answer to a look-up or connecting question (the kinds whose notes state key points)."""
    gradable = set(questions.loc[questions["kind"].isin(CORRECT_KINDS) & (questions["notes"].str.strip() != ""), "id"])
    return [Item(f"{r.qid}|{r.system}", r.qid, r.system) for r in answers.itertuples() if r.qid in gradable]


def correctness_prompt(question: str, notes: str, answer: str) -> str:
    return (f"Question:\n{question}\n\n---\nKey points a good answer contains (coach's note on the source "
            f"article):\n{notes}\n\n---\nAnswer:\n{answer}\n\n---\n"
            "coverage: how many of the key points does the answer convey (all, most, some, none)? "
            "contradicts: does it state anything that contradicts the key points? "
            "missing: the key points it leaves out, briefly. reason: one sentence.")


def grade_one(client, item: Item, question: str, notes: str, answer: str, ledger: SpendLedger,
              model: str = MODEL) -> dict:
    resp = client.beta.messages.create(
        model=model, max_tokens=8000, system=CORRECT_SYSTEM,
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": CORRECT_SCHEMA}},
        messages=[{"role": "user", "content": correctness_prompt(question, notes, answer)}],
    )
    ledger.record("correctness", resp.model, resp.usage.input_tokens, resp.usage.output_tokens)
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"refusal on {item.item_id}")
    data = json.loads(next(b.text for b in resp.content if b.type == "text"))
    return {"item_id": item.item_id, "qid": item.qid, "system": item.system, "coverage": data["coverage"],
            "score": COVERAGE_SCORE[data["coverage"]], "contradicts": data["contradicts"],
            "missing": data["missing"], "reason": data["reason"], "model_used": resp.model}


def run_correctness(items: list[Item], answers: pd.DataFrame, questions: pd.DataFrame, ledger: SpendLedger, client,
                    out: Path, *, yes: bool = False, limit: int | None = None, errors: Path | None = None) -> None:
    out, errors = Path(out), Path(errors or Path(out).with_name("correctness_errors.csv"))
    done = set() if not out.exists() else {json.loads(x)["item_id"] for x in out.read_text().splitlines() if x.strip()}
    ans = {(r.qid, r.system): r.answer for r in answers.itertuples()}
    q = questions.set_index("id")

    def job(it: Item):
        return lambda: grade_one(client, it, q.at[it.qid, "question"], q.at[it.qid, "notes"],
                                 ans[(it.qid, it.system)], ledger)

    todo = [(it.item_id, job(it)) for it in items if it.item_id not in done][:limit]
    _paid_loop(todo, out, ledger, errors, yes=yes, label="correctness", first_estimate=CORRECT_FIRST_CALL_USD)
