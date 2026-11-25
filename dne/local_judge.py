"""Second judge on Kaggle: a free open model from another family than the answer model
(Mistral Small 24B AWQ on vLLM), with exactly the Opus judge's system prompt, user prompt, four criteria and
JSON schema, and the same record shape — so judgments_local.jsonl lines up with judgments.jsonl pair by pair.

    python dne/local_judge.py --answers answers.parquet --questions questions.csv --out judgments_local.jsonl
                              --model <served name> [--concurrency 6] [--limit N] [--timeout 600]
                              [--stop-after-hours H]

Resumable (finished pair_ids are skipped; error rows are retried), fail-fast when the first 5 calls all fail,
progress + ETA after every call, no new calls after --stop-after-hours.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from dne.judge import CRITERIA, SCHEMA, SYSTEM, Pair, build_pairs, judge_prompt, to_system_winners  # noqa: E402

URL = "http://127.0.0.1:8000/v1/chat/completions"
FAIL_FAST_N = 5


def request_body(model: str, question: str, answer_a: str, answer_b: str) -> dict:
    return {"model": model, "temperature": 0, "max_tokens": 1500,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": judge_prompt(question, answer_a, answer_b)}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "verdict", "schema": SCHEMA, "strict": True}}}


def parse_record(pair: Pair, content: str, model: str) -> dict:
    """The model's JSON verdict → the Opus judge's record shape. ValueError on anything malformed."""
    try:
        data = json.loads(content)
        winners = {c: data[c]["winner"] for c in CRITERIA}
        reasons = {c: str(data[c].get("reason", "")) for c in CRITERIA}
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise ValueError(f"malformed verdict: {e.__class__.__name__}") from None
    if any(w not in ("A", "B", "tie") for w in winners.values()):
        raise ValueError(f"bad winner value in {winners}")
    return {"pair_id": pair.pair_id, "qid": pair.qid, "sys_a": pair.sys_a, "sys_b": pair.sys_b,
            "order": pair.order, "criterion_winners": winners, "system_winners": to_system_winners(pair, winners),
            "reasons": reasons, "model_used": model}


def todo_pairs(pairs: list[Pair], out: Path) -> list[Pair]:
    done = set()
    if Path(out).exists():
        for line in Path(out).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if not r.get("error"):
                    done.add(r["pair_id"])
    return [p for p in pairs if p.pair_id not in done]


def broken_start(finished: list[dict]) -> bool:
    """True when the first FAIL_FAST_N calls all failed — the model or server is broken, stop paying quota."""
    first = finished[:FAIL_FAST_N]
    return len(first) == FAIL_FAST_N and all(r.get("error") for r in first)


def _post(body: dict, timeout: float) -> str:
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--answers", type=Path, required=True)
    p.add_argument("--questions", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--stop-after-hours", type=float, default=None)
    a = p.parse_args()

    answers = pd.read_parquet(a.answers)
    questions = pd.read_csv(a.questions, dtype=str, keep_default_na=False)
    ans = {(r.qid, r.system): r.answer for r in answers.itertuples()}
    qtext = dict(zip(questions["id"], questions["question"]))
    todo = todo_pairs(build_pairs(answers, ["basic", "local", "global"]), a.out)[:a.limit]
    t0, lock, finished, stop = time.time(), threading.Lock(), [], threading.Event()
    print(f"[judge {0:6.0f}s] {len(todo)} pairs to judge with {a.model}, concurrency {a.concurrency}", flush=True)

    def one(pair: Pair) -> None:
        if stop.is_set():
            return
        if a.stop_after_hours and time.time() - t0 > a.stop_after_hours * 3600:
            stop.set()
            print(f"[judge {time.time() - t0:6.0f}s] TIME LIMIT {a.stop_after_hours} h — no new calls", flush=True)
            return
        shown = (pair.sys_a, pair.sys_b) if pair.order == 0 else (pair.sys_b, pair.sys_a)
        t = time.time()
        try:
            content = _post(request_body(a.model, qtext[pair.qid], ans[(pair.qid, shown[0])],
                                         ans[(pair.qid, shown[1])]), a.timeout)
            rec = parse_record(pair, content, a.model)
        except Exception as e:  # noqa: BLE001 — one bad call is an error row; a rerun retries it
            rec = {"pair_id": pair.pair_id, "qid": pair.qid, "error": f"{e.__class__.__name__}: {e}"[:300]}
        rec["latency_s"] = time.time() - t
        with lock:
            with a.out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            finished.append(rec)
            n, el = len(finished), time.time() - t0
            eta = el / n * (len(todo) - n)
            err = f" ERROR {rec['error']}" if rec.get("error") else ""
            print(f"[judge {el:6.0f}s] {n}/{len(todo)} {pair.pair_id}: {rec['latency_s']:.0f}s{err} | "
                  f"ETA {eta / 60:.0f} min", flush=True)
            if broken_start(finished) and not stop.is_set():
                stop.set()
                print(f"[judge {el:6.0f}s] FAIL-FAST: first {FAIL_FAST_N} calls all failed — stopping", flush=True)

    with ThreadPoolExecutor(a.concurrency) as ex:
        list(ex.map(one, todo))
    n_err = sum(1 for r in finished if r.get("error"))
    print(f"[judge {time.time() - t0:6.0f}s] done: {len(finished) - n_err} judged, {n_err} errors", flush=True)
    return 3 if stop.is_set() else 0


if __name__ == "__main__":
    sys.exit(main())
