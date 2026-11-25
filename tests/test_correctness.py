import json
from types import SimpleNamespace

import pandas as pd

from dne.judge import (COVERAGE_SCORE, correctness_items, correctness_prompt, run_correctness)
from dne.spend import SpendLedger


def _answers(qids, systems=("basic", "local", "global")):
    return pd.DataFrame([{"qid": q, "system": s, "answer": f"{s} answer to {q}"} for q in qids for s in systems])


QS = pd.DataFrame({
    "id": ["L1", "C1", "B1"],
    "kind": ["lookup", "connecting", "big_picture"],
    "question": ["Which points matter most?", "How do A and B connect?", "How has X changed?"],
    "notes": ["Roure 2014: 30-all and first advantage", "links A to B", "supports: 549 (2011), 412 (2012)"],
})


class FakeClient:
    def __init__(self, coverage="most", contradicts=False):
        self.calls, self.prompts = 0, []
        self.verdict = {"coverage": coverage, "contradicts": contradicts, "missing": "m", "reason": "r"}
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls += 1
        self.prompts.append(kw["messages"][0]["content"])
        return SimpleNamespace(model="claude-opus-5-5", stop_reason="end_turn",
                               content=[SimpleNamespace(type="text", text=json.dumps(self.verdict))],
                               usage=SimpleNamespace(input_tokens=1500, output_tokens=300))


def test_items_cover_lookup_and_connecting_only():
    items = correctness_items(_answers(["L1", "C1", "B1"]), QS)
    assert {i.qid for i in items} == {"L1", "C1"}
    assert len(items) == 6
    assert items[0].item_id == "L1|basic"


def test_prompt_has_notes_but_no_system_name():
    p = correctness_prompt("Which points matter most?", "Roure 2014: 30-all", "basic answer text")
    assert "30-all" in p and "basic answer text" in p
    assert "Plain RAG" not in p and "GraphRAG" not in p


def test_run_records_score_and_resumes(tmp_path):
    answers, out, led = _answers(["L1"]), tmp_path / "c.jsonl", SpendLedger(tmp_path / "l.csv")
    client = FakeClient(coverage="most", contradicts=True)
    items = correctness_items(answers, QS)
    run_correctness(items, answers, QS, led, client, out, yes=True)
    run_correctness(items, answers, QS, led, client, out, yes=True)
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert client.calls == 3 and len(rows) == 3
    assert rows[0]["score"] == COVERAGE_SCORE["most"] == 2
    assert rows[0]["contradicts"] is True and rows[0]["system"] == "basic"
    assert "30-all" in client.prompts[0]


def test_run_without_yes_makes_no_calls(tmp_path, capsys):
    client = FakeClient()
    answers = _answers(["L1"])
    run_correctness(correctness_items(answers, QS), answers, QS, SpendLedger(tmp_path / "l.csv"), client,
                    tmp_path / "c.jsonl", yes=False)
    assert client.calls == 0
    out = capsys.readouterr().out
    assert "--yes" in out and "3 calls" in out
