"""Answer the question set with graphrag basic / local / global search.

Runs inside the graphrag venv on the Kaggle box (started by kaggle/index/run_index.py, mode 'query').
Up to --concurrency queries run at once (throughput on 2×T4 is generation-bound), so
per-query tokens and LLM calls come from graphrag's own SearchResult counts, not vLLM counter deltas;
latency_s is measured under that load (the `concurrency` column says so). The vLLM counters still
give exact run totals, printed at the end.
Global search reads the community reports at --global-level (0 = root communities, the GraphRAG
paper's C0; level 2 took 45 min per question). Local search keeps graphrag's level 2.
One JSON line per (question, system) goes to --out as soon as it finishes; a rerun skips those.
Prints progress + ETA after every query and stops starting new ones when projected over budget.

    python query_worker.py --root R --questions Q.csv --out answers.jsonl --systems basic,local,global
                           [--per-kind N] [--budget-hours H] [--concurrency 3] [--global-level 0]
                           [--query-timeout S] [--stop-after-hours H]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402
import tiktoken  # noqa: E402
from graphrag.config.embeddings import entity_description_embedding, text_unit_text_embedding  # noqa: E402
from graphrag.config.load_config import load_config  # noqa: E402
from graphrag.query.factory import (get_basic_search_engine, get_global_search_engine,  # noqa: E402
                                    get_local_search_engine)
from graphrag.query.indexer_adapters import (read_indexer_communities, read_indexer_entities,  # noqa: E402
                                             read_indexer_relationships, read_indexer_reports,
                                             read_indexer_text_units)
from graphrag.utils.api import get_embedding_store, load_search_prompt  # noqa: E402
from graphrag_llm.utils import CompletionMessagesBuilder  # noqa: E402

from dne.progress import QUERY_BUDGET_GRACE_S, eta_seconds, over_budget, with_timeout  # noqa: E402
from dne.query_plan import broken_systems, plan_jobs, projection_line, sources_to_aids  # noqa: E402
from dne.vllm_metrics import parse_token_counters  # noqa: E402

RESPONSE_TYPE = "Multiple Paragraphs"  # graphrag's CLI default, the same for every system
LOCAL_LEVEL = 2  # graphrag's CLI default
T0 = time.time()


def say(msg: str) -> None:
    print(f"[query {time.time() - T0:6.0f}s] {msg}", flush=True)


def counters() -> dict[str, float]:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=30) as r:
            return parse_token_counters(r.read().decode())
    except Exception:  # noqa: BLE001 — totals only; must not stop the run
        return {"prompt_tokens": float("nan"), "generation_tokens": float("nan"), "requests": float("nan")}


def make_engine(system: str, cfg, t: dict[str, pd.DataFrame], global_level: int):
    """A fresh engine per query (the same calls graphrag.api makes), so concurrent queries share no state."""
    if system == "basic":
        return get_basic_search_engine(
            config=cfg, text_units=read_indexer_text_units(t["text_units"]),
            text_unit_embeddings=get_embedding_store(config=cfg.vector_store, embedding_name=text_unit_text_embedding),
            response_type=RESPONSE_TYPE, system_prompt=load_search_prompt(cfg.basic_search.prompt))
    if system == "local":
        return get_local_search_engine(
            config=cfg,
            reports=read_indexer_reports(t["community_reports"], t["communities"], LOCAL_LEVEL),
            text_units=read_indexer_text_units(t["text_units"]),
            entities=read_indexer_entities(t["entities"], t["communities"], LOCAL_LEVEL),
            relationships=read_indexer_relationships(t["relationships"]), covariates={"claims": []},
            description_embedding_store=get_embedding_store(config=cfg.vector_store,
                                                            embedding_name=entity_description_embedding),
            response_type=RESPONSE_TYPE, system_prompt=load_search_prompt(cfg.local_search.prompt))
    if system == "global":
        return get_global_search_engine(
            cfg,
            reports=read_indexer_reports(t["community_reports"], t["communities"], community_level=global_level,
                                         dynamic_community_selection=False),
            entities=read_indexer_entities(t["entities"], t["communities"], community_level=global_level),
            communities=read_indexer_communities(t["communities"], t["community_reports"]),
            response_type=RESPONSE_TYPE, dynamic_community_selection=False,
            map_system_prompt=load_search_prompt(cfg.global_search.map_prompt),
            reduce_system_prompt=load_search_prompt(cfg.global_search.reduce_prompt),
            general_knowledge_inclusion_prompt=load_search_prompt(cfg.global_search.knowledge_prompt))
    raise ValueError(system)


async def basic_answer(engine, query: str) -> SimpleNamespace:
    """graphrag 3.3's BasicSearch.search with its bug fixed: it calls model.completion_async(stream=True)
    without `await`, iterates the coroutine, catches the TypeError and returns "".
    Same context, prompt, model call and token counting as graphrag's own code; stream_search awaits it."""
    ctx = engine.context_builder.build_context(query=query, conversation_history=None, **engine.context_builder_params)
    prompt = engine.system_prompt.format(context_data=ctx.context_chunks, response_type=engine.response_type)
    msgs = CompletionMessagesBuilder().add_system_message(prompt).add_user_message(query).build()
    stream = await engine.model.completion_async(messages=msgs, stream=True, **engine.model_params)
    response = "".join([c.choices[0].delta.content or "" async for c in stream])
    return SimpleNamespace(response=response, context_data=ctx.context_records, llm_calls=1,
                           prompt_tokens=len(engine.tokenizer.encode(prompt)),
                           output_tokens=len(engine.tokenizer.encode(response)), llm_calls_categories={"response": 1})


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--questions", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--systems", default="basic,local,global")
    p.add_argument("--per-kind", type=int, default=None)
    p.add_argument("--budget-hours", type=float, default=6.0)
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--global-level", type=int, default=0)
    p.add_argument("--query-timeout", type=float, default=1500.0,
                   help="abandon one query after this many seconds (error row; a rerun retries it)")
    p.add_argument("--stop-after-hours", type=float, default=None,
                   help="start no new query after this (quota-limited session); running ones finish")
    a = p.parse_args()

    cfg = load_config(root_dir=a.root)
    out_dir = a.root / "output"
    tables = {n: pd.read_parquet(out_dir / f"{n}.parquet") for n in
              ("entities", "communities", "community_reports", "text_units", "relationships", "documents")}
    questions = pd.read_csv(a.questions, dtype=str, keep_default_na=False)
    text_of = dict(zip(questions.id, questions.question))
    systems = a.systems.split(",")
    done = set()
    if a.out.exists():
        done = {(r["qid"], r["system"]) for r in map(json.loads, a.out.read_text().splitlines()) if not r.get("error")}
    jobs = plan_jobs(questions, systems, done, per_kind=a.per_kind)
    n_q = len({q for q, _ in jobs})
    say(f"{len(jobs)} queries ({n_q} questions × {systems}), {len(done)} already done, budget {a.budget_hours} h, "
        f"concurrency {a.concurrency}, global level {a.global_level}")
    enc = tiktoken.get_encoding("o200k_base")
    budget_s, t_start, c_start = a.budget_hours * 3600, time.time(), counters()
    per_system = {s: 0.0 for s in systems}
    sem, stop, n_done, finished = asyncio.Semaphore(a.concurrency), asyncio.Event(), 0, []

    async def run(qid: str, system: str) -> None:
        nonlocal n_done
        async with sem:
            if stop.is_set():
                return
            if a.stop_after_hours and time.time() - t_start > a.stop_after_hours * 3600:
                stop.set()
                say(f"TIME LIMIT: {a.stop_after_hours} h reached — no new queries; rerun resumes from {a.out.name}")
                return
            t = time.time()
            row = {"qid": qid, "system": system, "concurrency": a.concurrency,
                   "community_level": a.global_level if system == "global" else LOCAL_LEVEL}
            try:
                engine = make_engine(system, cfg, tables, a.global_level)
                res = await with_timeout(basic_answer(engine, text_of[qid]) if system == "basic"
                                         else engine.search(text_of[qid]), a.query_timeout)
                answer = res.response if isinstance(res.response, str) else json.dumps(res.response)
                row.update(answer=answer, answer_tokens=len(enc.encode(answer)),
                           retrieved_aids=sources_to_aids(res.context_data, tables["text_units"], tables["documents"]),
                           prompt_tokens=res.prompt_tokens, generation_tokens=res.output_tokens,
                           llm_calls=res.llm_calls, llm_calls_categories=res.llm_calls_categories or {},
                           error="" if answer.strip() else "empty answer")
            except Exception as e:  # noqa: BLE001 — one failed query must not end the run; rerun picks it up
                row.update(answer="", answer_tokens=0, retrieved_aids=[], prompt_tokens=0, generation_tokens=0,
                           llm_calls=0, error=f"{e.__class__.__name__}: {e}"[:500])
            dt = time.time() - t
            row["latency_s"] = dt
            per_system[system] += dt
            with a.out.open("a") as f:
                f.write(json.dumps(row) + "\n")
            n_done += 1
            finished.append(row)
            broken = broken_systems(finished)
            if broken and not stop.is_set():
                stop.set()
                say(f"FAIL-FAST: first 3 answers of {broken} all failed — stopping the run (see query_worker.log)")
            elapsed = time.time() - t_start
            eta = eta_seconds(n_done, len(jobs), elapsed) or 0
            say(f"{n_done}/{len(jobs)} {qid} {system}: {dt:.0f}s, {row['llm_calls']} calls, "
                f"{row['prompt_tokens']}+{row['generation_tokens']} tok, {row['answer_tokens']} answer tok"
                f"{' ERROR ' + row['error'] if row['error'] else ''} | ETA {eta / 60:.0f} min")
            if not stop.is_set() and over_budget(n_done, len(jobs), elapsed, budget_s, grace_s=QUERY_BUDGET_GRACE_S):
                stop.set()
                say(f"WATCHDOG: projected {(elapsed + eta) / 3600:.1f} h > budget {a.budget_hours} h — no new "
                    f"queries; running ones finish; rerun resumes from {a.out.name}")

    await asyncio.gather(*(run(q, s) for q, s in jobs))
    c_end = counters()
    say("vLLM totals this run: " + ", ".join(f"{k} {c_end[k] - c_start[k]:.0f}" for k in c_end))
    if stop.is_set():
        return 3
    if a.per_kind and n_q:
        say("BENCHMARK " + projection_line(n_q, time.time() - t_start, len(questions), per_system))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))  # one event loop for all queries (graphrag's HTTP clients outlive a query)
