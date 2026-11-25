"""A second, free judge (Mistral Small 24B on Kaggle) — same prompt, criteria and record shape
as the Opus judge, so the two can be compared pair by pair."""
import json

import pytest

from dne.judge import CRITERIA, SCHEMA, SYSTEM, Pair
from dne.local_judge import broken_start, parse_record, request_body, todo_pairs

PAIR = Pair(pair_id="A01|basic|local|1", qid="A01", sys_a="basic", sys_b="local", order=1)


def _verdict(**w):
    return json.dumps({c: {"winner": w.get(c, "tie"), "reason": "r"} for c in CRITERIA})


def test_request_uses_the_opus_prompt_and_schema():
    body = request_body("m", "Q?", "ans one", "ans two")
    assert body["model"] == "m" and body["temperature"] == 0
    assert body["messages"][0] == {"role": "system", "content": SYSTEM}
    user = body["messages"][1]["content"]
    assert "Answer A:\nans one" in user and "Answer B:\nans two" in user
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA
    assert "basic" not in user and "local" not in user  # the judge never sees system names


def test_parse_record_maps_display_letters_back_to_systems():
    rec = parse_record(PAIR, _verdict(comprehensiveness="A", diversity="B"), "mistral")
    # order 1: answer A was sys_b (local), answer B was sys_a (basic)
    assert rec["system_winners"]["comprehensiveness"] == "local"
    assert rec["system_winners"]["diversity"] == "basic"
    assert rec["system_winners"]["directness"] == "tie"
    assert rec["pair_id"] == PAIR.pair_id and rec["model_used"] == "mistral"
    assert set(rec) >= {"qid", "sys_a", "sys_b", "order", "criterion_winners", "reasons"}


def test_parse_record_rejects_broken_json():
    with pytest.raises(ValueError):
        parse_record(PAIR, '{"comprehensiveness": {"winner": "A"', "m")
    with pytest.raises(ValueError):
        parse_record(PAIR, json.dumps({"comprehensiveness": {"winner": "C", "reason": ""}}), "m")


def test_todo_skips_finished_pairs_but_retries_errors(tmp_path):
    out = tmp_path / "j.jsonl"
    p2 = Pair("A01|basic|local|0", "A01", "basic", "local", 0)
    out.write_text(json.dumps({"pair_id": PAIR.pair_id}) + "\n"
                   + json.dumps({"pair_id": p2.pair_id, "error": "timeout"}) + "\n")
    assert todo_pairs([PAIR, p2], out) == [p2]


def test_broken_start_stops_when_the_first_answers_all_fail():
    assert broken_start([{"error": "x"}] * 5)
    assert not broken_start([{"error": "x"}] * 4 + [{"system_winners": {}}])
    assert not broken_start([{"error": "x"}] * 3)  # too early to tell
