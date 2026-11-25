"""Run a scripted demo dialogue through basic / local / global search (brief §13.1: conversational agent).

Runs inside the graphrag venv on the Kaggle box (started by kaggle/index/run_index.py, mode 'chat').
Each system gets the same dialogues (user turns in order; '---' starts a new conversation): a follow-up is first rewritten into a
standalone question by the served LLM (dne.chat.CONDENSE_PROMPT), then answered by that system with the same
engines, settings and context budget as the evaluated answers (dne.query_worker.make_engine). The three
systems run side by side; turns within a system run one after another.
Self-abort: no turn starts after --stop-after-hours (checked before every turn) and one turn is abandoned after
--query-timeout; the first turn's projection is logged only (24/11: warm-up made it 3-5x too pessimistic). One JSON line per turn goes to --out; chat_transcript.md at the end.

    python chat_worker.py --root R --dialogue dialogue.txt --out chat.jsonl --model M
                          [--systems basic,local,global] [--global-level 0] [--stop-after-hours 1.0]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402
import tiktoken  # noqa: E402
from graphrag.config.load_config import load_config  # noqa: E402

from dne.chat import CONDENSE_PROMPT, ChatSession, format_history, load_dialogues, transcript_markdown  # noqa: E402
from dne.progress import eta_seconds, with_timeout  # noqa: E402
from dne.query_plan import sources_to_aids  # noqa: E402
from dne.query_worker import basic_answer, make_engine  # noqa: E402

T0 = time.time()


def say(msg: str) -> None:
    print(f"[chat {time.time() - T0:6.0f}s] {msg}", flush=True)


def condense(model: str, question: str, history: list[dict]) -> str:
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": 200, "messages": [
        {"role": "user", "content": CONDENSE_PROMPT.format(history=format_history(history), question=question)}]})
    req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", data=body.encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        text = json.loads(r.read())["choices"][0]["message"]["content"].strip()
    return text or question


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--dialogue", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--systems", default="basic,local,global")
    p.add_argument("--global-level", type=int, default=0)
    p.add_argument("--query-timeout", type=float, default=900.0)
    p.add_argument("--stop-after-hours", type=float, default=1.0)
    a = p.parse_args()

    cfg = load_config(root_dir=a.root)
    tables = {n: pd.read_parquet(a.root / "output" / f"{n}.parquet") for n in
              ("entities", "communities", "community_reports", "text_units", "relationships", "documents")}
    dialogues = load_dialogues(a.dialogue)
    n_turns = sum(len(d) for d in dialogues)
    systems = a.systems.split(",")
    limit_s = a.stop_after_hours * 3600
    enc = tiktoken.get_encoding("o200k_base")
    say(f"{len(dialogues)} dialogues, {n_turns} turns × {systems}, global level {a.global_level}, stop after {a.stop_after_hours} h")
    rows: list[dict] = []

    async def run_system(system: str) -> None:
        last: dict = {}

        async def search(query: str, _history: list[dict]) -> str:
            engine = make_engine(system, cfg, tables, a.global_level)
            res = await with_timeout(basic_answer(engine, query) if system == "basic" else engine.search(query),
                                     a.query_timeout)
            answer = res.response if isinstance(res.response, str) else json.dumps(res.response)
            last.update(llm_calls=res.llm_calls, prompt_tokens=res.prompt_tokens,
                        generation_tokens=res.output_tokens, answer_tokens=len(enc.encode(answer)),
                        retrieved_aids=sources_to_aids(res.context_data, tables["text_units"], tables["documents"]))
            return answer

        async def rewrite(question: str, history: list[dict]) -> str:
            return await asyncio.to_thread(condense, a.model, question, history)

        t_sys, n_done = time.time(), 0
        for d, turns in enumerate(dialogues, start=1):
            chat = ChatSession(search, rewrite_fn=rewrite)  # a fresh conversation per dialogue
            for i, user in enumerate(turns, start=1):
                if time.time() - T0 > limit_s:
                    say(f"{system}: TIME LIMIT {a.stop_after_hours} h reached — stopping before dialogue {d} turn {i}")
                    return
                last.clear()
                t = time.time()
                row = {"dialogue": d, "system": system, "turn": i, "user": user}
                try:
                    answer = await chat.aask(user)
                    row.update(standalone=chat.standalone[-1], answer=answer, error="" if answer.strip() else "empty")
                except Exception as e:  # noqa: BLE001 — a failed turn ends this dialogue for this system only
                    row.update(standalone=chat.standalone[-1] if len(chat.standalone) == i else "", answer="",
                               error=f"{e.__class__.__name__}: {e}"[:500])
                row.update(last, latency_s=time.time() - t)
                rows.append(row)
                with a.out.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                n_done += 1
                eta = eta_seconds(n_done, n_turns, time.time() - t_sys) or 0
                say(f"{system} dialogue {d} turn {i}/{len(turns)}: {row['latency_s']:.0f}s, "
                    f"{row.get('llm_calls', 0)} calls{' ERROR ' + row['error'] if row['error'] else ''} "
                    f"| ETA {eta / 60:.0f} min")
                if n_done == 1:  # log only: the first turn includes warm-up and was 3-5x slower than later
                    # turns on 24/11 (run 1 stopped global wrongly on it); the time limit below each turn stops the run
                    projected = time.time() - T0 + eta
                    say(f"BENCHMARK {system}: first turn {row['latency_s']:.0f}s → projected end {projected / 60:.0f} min "
                        f"(limit {limit_s / 60:.0f} min; checked before every turn)")
                if row["error"]:
                    say(f"{system}: dialogue {d} stopped (follow-ups depend on this turn)")
                    break

    await asyncio.gather(*(run_system(s) for s in systems))
    (a.out.parent / "chat_transcript.md").write_text("# Demo dialogue\n\n" + transcript_markdown(rows))
    n_ok = sum(1 for r in rows if not r["error"])
    say(f"done: {n_ok}/{n_turns * len(systems)} turns answered; transcript in chat_transcript.md")
    return 0 if n_ok == n_turns * len(systems) else 3


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
