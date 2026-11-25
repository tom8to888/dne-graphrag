import json
from types import SimpleNamespace

import pandas as pd
import pytest

from dne.judge import CRITERIA, Pair, build_pairs, run_judge, to_system_winners
from dne.spend import SpendLedger


def _answers(qids, systems=("basic", "local", "global")):
    return pd.DataFrame([{"qid": q, "system": s, "answer": f"{s} answer to {q}"} for q in qids for s in systems])


def _questions(qids):
    return pd.DataFrame({"id": qids, "question": [f"question {q}?" for q in qids]})


class FakeClient:
    """Mimics client.beta.messages.create; verdict 'A' on every criterion; ~$0.05 per call on Opus."""

    def __init__(self, fail_on=()):
        self.calls = 0
        self.fail_on = set(fail_on)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls += 1
        if self.calls in self.fail_on:
            raise RuntimeError("boom")
        verdict = {c: {"winner": "A", "reason": "r"} for c in CRITERIA}
        return SimpleNamespace(model="claude-opus-5-5", stop_reason="end_turn",
                               content=[SimpleNamespace(type="text", text=json.dumps(verdict))],
                               usage=SimpleNamespace(input_tokens=5000, output_tokens=1500))


def test_build_pairs_counts():
    pairs = build_pairs(_answers(["q1", "q2"]), ["basic", "local", "global"])
    assert len(pairs) == 2 * 3 * 2
    assert {p.order for p in pairs} == {0, 1}


def test_to_system_winners_swapped():
    p = Pair(pair_id="q1|basic|global|1", qid="q1", sys_a="basic", sys_b="global", order=1)
    # order 1 shows sys_b ('global') as answer A
    assert to_system_winners(p, {"comprehensiveness": "A", "diversity": "B", "directness": "tie"}) == {
        "comprehensiveness": "global", "diversity": "basic", "directness": "tie"}


def test_run_judge_resume_no_double_spend(tmp_path):
    pairs = build_pairs(_answers(["q1"], ("basic", "global")), ["basic", "global"])
    led, out, client = SpendLedger(tmp_path / "l.csv"), tmp_path / "j.jsonl", FakeClient()
    run_judge(pairs, _answers(["q1"], ("basic", "global")), _questions(["q1"]), led, client, out, yes=True)
    run_judge(pairs, _answers(["q1"], ("basic", "global")), _questions(["q1"]), led, client, out, yes=True)
    assert client.calls == 2
    assert len(out.read_text().splitlines()) == 2
    assert led.total() == pytest.approx(2 * (5000 * 4 + 1500 * 20) / 1e6)


def test_run_judge_stops_at_cap(tmp_path, capsys):
    led = SpendLedger(tmp_path / "l.csv", cap_usd=20.0)
    led.record("seed", "claude-opus-5-5", 4_997_500, 0)  # $19.99
    client = FakeClient()
    pairs = build_pairs(_answers(["q1"], ("basic", "global")), ["basic", "global"])
    run_judge(pairs, _answers(["q1"], ("basic", "global")), _questions(["q1"]), led, client,
              tmp_path / "j.jsonl", yes=True)
    assert client.calls == 0
    assert "cap" in capsys.readouterr().out


def test_run_judge_api_error_continues(tmp_path):
    answers = _answers(["q1", "q2"], ("basic", "global"))
    pairs = build_pairs(answers, ["basic", "global"])  # 4 pairs
    client = FakeClient(fail_on={2})
    run_judge(pairs, answers, _questions(["q1", "q2"]), SpendLedger(tmp_path / "l.csv"), client,
              tmp_path / "j.jsonl", yes=True, errors=tmp_path / "e.csv")
    assert len((tmp_path / "j.jsonl").read_text().splitlines()) == 3
    assert pairs[1].pair_id in (tmp_path / "e.csv").read_text()


def test_run_judge_without_yes_makes_no_calls(tmp_path, capsys):
    client = FakeClient()
    pairs = build_pairs(_answers(["q1"], ("basic", "global")), ["basic", "global"])
    run_judge(pairs, _answers(["q1"], ("basic", "global")), _questions(["q1"]), SpendLedger(tmp_path / "l.csv"),
              client, tmp_path / "j.jsonl", yes=False)
    assert client.calls == 0
    assert "--yes" in capsys.readouterr().out
