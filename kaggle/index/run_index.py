"""Kaggle GPU kernel: serve an open LLM + embedder with vLLM, then run one mode from run_config.json:
'smoke' (a quick end-to-end test), 'index' (build the GraphRAG index), 'query' (answer the question set on a
finished index), 'chat' (the demo dialogues) or 'judge' (the second, local LLM judge).

kaggle/index/run_index.py and kaggle/query/run_index.py are the same file: `kaggle kernels push` takes one
folder per kernel, each with its own kernel-metadata.json. tests/test_kaggle_bundle.py keeps them identical.

Reads run_config.json from the private code dataset (dne-graphrag-code) and the article texts from
the private text dataset (itf-ccsr-text). Everything worth keeping goes to /kaggle/working.
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

OUT = Path("/kaggle/working")
LOG = OUT / "logs"
LOG.mkdir(parents=True, exist_ok=True)


def find_dir(marker: str) -> Path:
    for root, _dirs, files in os.walk("/kaggle/input"):
        if marker in files:
            return Path(root)
    raise FileNotFoundError(marker)


CODE = find_dir("run_config.json")
CFG = json.loads((CODE / "run_config.json").read_text())
sys.path.insert(0, str(CODE))
from dne.power import energy_kwh  # noqa: E402
from dne.progress import expected_extract_calls, over_budget, progress_line, stalled, worker_deadline_s  # noqa: E402
from dne.resume import (find_cache_archive, find_output, pack_cache, pack_output, restore_cache,  # noqa: E402
                        restore_output, session_time_up)
from dne.vllm_metrics import parse_gauges, parse_token_counters  # noqa: E402

FULL_ARTICLES = 541  # usable corpus size, for the live full-run projection
WATCH_EVERY_S = 120
STALL_S = 1200  # no new LLM result for 20 min → stop and pack the cache

T0 = time.time()


def say(msg):
    line = f"[{time.time() - T0:7.0f}s] {msg}"
    print(line, flush=True)
    with (LOG / "run.log").open("a") as f:
        f.write(line + "\n")


def sh(cmd, log_name, **kw):
    with (LOG / log_name).open("a") as f:
        return subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, **kw).returncode


_last_counters = {"prompt_tokens": 0.0, "generation_tokens": 0.0, "requests": 0.0}


def counters():
    """vLLM token counters; the last known values if the server is gone (a dead server must not stop
    this script before it saves the cache)."""
    global _last_counters
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=30) as r:
            _last_counters = parse_token_counters(r.read().decode())
    except Exception as e:  # noqa: BLE001
        say(f"warning: vLLM metrics unavailable ({e.__class__.__name__}); using last known counters")
    return _last_counters


def gauges():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=30) as r:
            return parse_gauges(r.read().decode())
    except Exception:  # noqa: BLE001 — a missed sample must not kill the watchdog
        return {}


def post(port, path, body, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# 1. environments: vLLM and graphrag in separate venvs (their pins clash)
say(f"config {CFG}")
sh("set -x; python -m pip install uv && python -m uv --version && env | grep -iE '^(uv_|pip_|virtual_env|python)' || true",
   "install.log")
UV = f"{sys.executable} -m uv"
VLLM_PY, GR_BIN = "/tmp/venv_vllm/bin/python", "/tmp/venv_gr/bin"


def make_env(path: str, package: str, check: str) -> None:
    t = time.time()
    rc = sh(f"set -x; {UV} venv {path} --python 3.12 && {UV} pip install --python {path}/bin/python {package}",
            "install.log")
    ok = sh(f"set -x; {path}/bin/python -c '{check}'", "install.log")
    say(f"{package}: install rc={rc}, import check rc={ok}, {time.time() - t:.0f}s")
    if ok != 0:
        sys.exit(f"{package} did not install — see logs/install.log")


make_env("/tmp/venv_vllm", f"vllm=={CFG['vllm']}", "import vllm; print(vllm.__version__)")
if CFG["mode"] in ("index", "query", "chat"):
    make_env("/tmp/venv_gr", f"graphrag=={CFG['graphrag']}", "import graphrag; print(graphrag.__name__)")

# 2. servers
t = time.time()
rc = sh(f"bash {CODE}/serve.sh {VLLM_PY} '{CFG['llm_model']}' '{CFG['embed_model']}' {LOG}", "serve.log")
say(f"serve rc={rc} after {time.time() - t:.0f}s")
if rc != 0:
    sys.exit(f"servers failed — see logs/serve.log, logs/vllm_llm.log, logs/vllm_embed.log")

info = {"started": datetime.now(timezone.utc).isoformat(), **CFG,
        "gpus": subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True,
                               capture_output=True, text=True).stdout.strip().splitlines()}

if CFG["mode"] == "smoke":
    # 3a. one JSON chat, one embedding, then 32 concurrent requests for throughput
    r = post(8000, "/v1/chat/completions", {"model": CFG["llm_model"], "max_tokens": 200, "messages": [
        {"role": "user", "content": "List three tennis drills for the backhand slice as JSON: {\"drills\": [...]}"}]})
    say("chat: " + r["choices"][0]["message"]["content"][:300].replace("\n", " "))
    e = post(8001, "/v1/embeddings", {"model": CFG["embed_model"], "input": ["split step timing"]})
    say(f"embedding dim {len(e['data'][0]['embedding'])}")
    c0, t = counters(), time.time()
    prompt = "Explain, in about 300 words, how a club player can improve their second serve. " * 3
    with ThreadPoolExecutor(32) as ex:
        list(ex.map(lambda _: post(8000, "/v1/chat/completions", {
            "model": CFG["llm_model"], "max_tokens": 400, "messages": [{"role": "user", "content": prompt}]}),
            range(32)))
    c1, dt = counters(), time.time() - t
    info["smoke"] = {"seconds": dt, "gen_tokens": c1["generation_tokens"] - c0["generation_tokens"],
                     "prompt_tokens": c1["prompt_tokens"] - c0["prompt_tokens"],
                     "gen_tok_per_s": (c1["generation_tokens"] - c0["generation_tokens"]) / dt}
    say(f"throughput: {info['smoke']}")
elif CFG["mode"] == "query":
    # 3c. answer the question set with basic / local / global search on the finished index
    root = OUT / "graphrag"
    root.mkdir(parents=True, exist_ok=True)
    src = find_output(Path("/kaggle/input"))
    if src is None:
        sys.exit("query mode needs the index dataset (output.tar.gz or unpacked output/) in /kaggle/input")
    say(f"index: restored {restore_output(src, root)} output tables from {src}")
    shutil.copy2(CODE / "settings.yaml", root / "settings.yaml")
    shutil.copytree(CODE / "prompts", root / "prompts", dirs_exist_ok=True)
    answers = OUT / "answers.jsonl"
    for dirpath, _d, files in os.walk("/kaggle/input"):  # an earlier session's answers (dataset) → resume
        if not CFG.get("resume_answers", True):
            say("fresh run: earlier answers dataset ignored")
            if (CODE / "answers_resume.jsonl").exists():  # this fresh run's own partial answers, shipped in the code
                shutil.copy2(CODE / "answers_resume.jsonl", answers)
                say(f"resume: {len(answers.read_text().splitlines())} answers of this run from the code bundle")
            break
        if "answers.jsonl" in files:
            shutil.copy2(Path(dirpath) / "answers.jsonl", answers)
            say(f"resume: {len(answers.read_text().splitlines())} earlier answers from {dirpath}")
            break
    cmd = (f"exec {GR_BIN}/python -u {CODE}/dne/query_worker.py --root {root} --questions {CODE}/questions.csv "
           f"--out {answers} --systems {','.join(CFG['systems'])} --budget-hours {CFG['budget_hours']} "
           f"--concurrency {CFG.get('query_concurrency', 3)} --global-level {CFG.get('global_level', 0)}")
    if CFG.get("per_kind"):
        cmd += f" --per-kind {CFG['per_kind']}"
    if CFG.get("query_stop_hours"):
        cmd += f" --stop-after-hours {CFG['query_stop_hours']}"
    q_timeout = CFG.get("query_timeout_s", 1500.0)
    cmd += f" --query-timeout {q_timeout}"
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = worker_deadline_s(CFG.get("query_stop_hours"), q_timeout)
    if deadline:  # last line of defence: answers are appended per query, so a kill loses only the running ones
        def _kill():
            say(f"WATCHDOG: query worker still running after {deadline / 3600:.2f} h — killing it")
            proc.kill()
        killer = threading.Timer(deadline, _kill)
        killer.daemon = True
        killer.start()
        say(f"watchdog: query worker hard stop at {deadline / 3600:.2f} h, one query max {q_timeout:.0f} s")
    with (LOG / "query_worker.log").open("w") as f:
        for line in proc.stdout:
            f.write(line)
            if line.startswith("[query"):  # progress, ETA, watchdog, benchmark projection
                say(line.rstrip())
    info["query_rc"] = proc.wait()
    say(f"query worker rc={info['query_rc']}")
    if answers.exists():
        import pandas as pd
        rows = pd.DataFrame([json.loads(x) for x in answers.read_text().splitlines()])
        rows = rows.drop_duplicates(["qid", "system"], keep="last")  # a rerun's retry replaces the failed row
        rows.to_parquet(OUT / "answers.parquet", index=False)
        say(f"answers.parquet: {len(rows)} rows, {int((rows.error != '').sum())} with errors")
elif CFG["mode"] == "chat":
    # 3e. scripted demo dialogue (brief: conversational agent): every system answers the same user turns
    root = OUT / "graphrag"
    root.mkdir(parents=True, exist_ok=True)
    src = find_output(Path("/kaggle/input"))
    if src is None:
        sys.exit("chat mode needs the index dataset (output.tar.gz or unpacked output/) in /kaggle/input")
    say(f"index: restored {restore_output(src, root)} output tables from {src}")
    shutil.copy2(CODE / "settings.yaml", root / "settings.yaml")
    shutil.copytree(CODE / "prompts", root / "prompts", dirs_exist_ok=True)
    stop_h = CFG.get("query_stop_hours") or 1.0
    cmd = (f"exec {GR_BIN}/python -u {CODE}/dne/chat_worker.py --root {root} --dialogue {CODE}/dialogue.txt "
           f"--out {OUT / 'chat.jsonl'} --model '{CFG['llm_model']}' --systems {','.join(CFG['systems'])} "
           f"--global-level {CFG.get('global_level', 0)} --query-timeout {CFG.get('query_timeout_s', 900)} "
           f"--stop-after-hours {stop_h}")
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = worker_deadline_s(stop_h, CFG.get("query_timeout_s", 900))
    killer = threading.Timer(deadline, lambda: (say("WATCHDOG: chat worker still running — killing it"), proc.kill()))
    killer.daemon = True
    killer.start()
    say(f"watchdog: chat worker hard stop at {deadline / 3600:.2f} h")
    with (LOG / "chat_worker.log").open("w") as f:
        for line in proc.stdout:
            f.write(line)
            if line.startswith("[chat"):
                say(line.rstrip())
    info["chat_rc"] = proc.wait()
    say(f"chat worker rc={info['chat_rc']}")
elif CFG["mode"] == "judge":
    # 3d. second judge: the Opus judge's prompt + schema on the served open model, over the finished answers
    jout = OUT / "judgments_local.jsonl"
    cmd = (f"exec {sys.executable} -u {CODE}/dne/local_judge.py --answers {CODE}/answers.parquet "
           f"--questions {CODE}/questions.csv --out {jout} --model '{CFG['llm_model']}' "
           f"--concurrency {CFG.get('judge_concurrency', 6)} --timeout {CFG.get('query_timeout_s', 600)}")
    if CFG.get("judge_limit"):
        cmd += f" --limit {CFG['judge_limit']}"
    if CFG.get("query_stop_hours"):
        cmd += f" --stop-after-hours {CFG['query_stop_hours']}"
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = worker_deadline_s(CFG.get("query_stop_hours"), CFG.get("query_timeout_s", 600))
    if deadline:
        killer = threading.Timer(deadline, lambda: (say("WATCHDOG: judge still running — killing it"), proc.kill()))
        killer.daemon = True
        killer.start()
        say(f"watchdog: judge hard stop at {deadline / 3600:.2f} h")
    with (LOG / "judge.log").open("w") as f:
        for line in proc.stdout:
            f.write(line)
            if line.startswith("[judge"):
                say(line.rstrip())
    info["judge_rc"] = proc.wait()
    say(f"judge rc={info['judge_rc']}")
else:
    # 3b. GraphRAG index over the first n_articles usable texts (0 = all)
    TEXT = find_dir("articles.jsonl")
    root = OUT / "graphrag"
    (root / "input").mkdir(parents=True, exist_ok=True)
    texts = sorted((json.loads(line) for line in (TEXT / "articles.jsonl").read_text().splitlines()),
                   key=lambda r: r["aid"])
    if CFG.get("aids"):
        wanted = set(CFG["aids"])
        texts = [r for r in texts if r["aid"] in wanted]
        if len(texts) != len(wanted):
            sys.exit(f"{len(wanted) - len(texts)} requested articles are not in the text dataset")
    elif CFG["n_articles"]:
        texts = texts[: CFG["n_articles"]]
    for r in texts:
        (root / "input" / f"{r['aid']}.txt").write_text(r["text"])
    shutil.copy2(CODE / "settings.yaml", root / "settings.yaml")
    shutil.copytree(CODE / "prompts", root / "prompts", dirs_exist_ok=True)
    say(f"indexing {len(texts)} articles")
    restored = 0
    if CFG.get("resume"):
        archive = find_cache_archive(Path("/kaggle/input"))
        if archive is None:
            sys.exit("resume=true but no cache (cache.tar.gz or unpacked cache/) in /kaggle/input")
        restored = restore_cache(archive, root)
        say(f"resume: restored {restored} cached extraction calls from {archive}")
    if CFG.get("restore_output"):  # finish a stopped index: earlier tables in, only the missing workflows run
        src = find_output(Path("/kaggle/input"))
        if src is None:
            sys.exit("restore_output=true but no output (output.tar.gz or unpacked output/) in /kaggle/input")
        say(f"resume: restored {restore_output(src, root)} output tables from {src}")
    stop_after_h = CFG.get("stop_after_hours", 11.0)
    try:
        import yaml
        st = yaml.safe_load((root / "settings.yaml").read_text())
        chunk, overlap = st["chunking"]["size"], st["chunking"]["overlap"]
        gleanings = st["extract_graph"].get("max_gleanings", 1)
    except Exception:  # noqa: BLE001
        st, chunk, overlap, gleanings = None, 1200, 100, 1
    total_calls = expected_extract_calls([r["text"] for r in texts], chunk, overlap, gleanings)
    budget_s = CFG.get("budget_hours", 9.0) * 3600
    say(f"watchdog: ~{total_calls} extraction calls expected, budget {budget_s / 3600:.1f} h, "
        f"progress every {WATCH_EVERY_S} s")
    stages, last, c_last = [], time.time(), counters()
    # exec: terminate() must reach graphrag itself, not only the shell
    proc = subprocess.Popen(f"exec {GR_BIN}/graphrag index --root {root} --verbose", shell=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    stop, aborted = threading.Event(), {}
    extract_done = threading.Event()  # set from graphrag's own "Workflow complete: extract_graph" line
    workflows = (st.get("workflows") if isinstance(st, dict) else None) or []
    if workflows and "extract_graph" not in workflows:
        extract_done.set()  # extraction is not part of this run: no extraction budget or ETA
        say(f"running only workflows: {workflows}")

    def watch():
        t_start, cache, t_extract = time.time(), root / "cache" / "extract_graph", None
        n_results, last_change = -1, time.time()
        while not stop.wait(WATCH_EVERY_S):
            # any cache entry (extraction, summaries, reports, embeddings) counts as progress
            n_now = sum(1 for p in (root / "cache").rglob("*") if p.is_file()) if (root / "cache").exists() else 0
            if n_now != n_results:
                n_results, last_change = n_now, time.time()
            elif stalled(last_change, time.time(), STALL_S):
                aborted["reason"] = f"stalled: no new LLM result for {STALL_S // 60} min — resume in a new session"
                say(f"WATCHDOG: {aborted['reason']} — stopping graphrag, cache kept")
                proc.terminate()
                return
            # count only this session's calls, so the ETA is not flattered by restored ones
            done = (sum(1 for _ in cache.iterdir()) if cache.exists() else 0) - restored
            todo = total_calls - restored
            elapsed = time.time() - t_start
            # the call count is only an estimate (v7: 262 real vs 268 expected) — trust graphrag's own stage line
            if extract_done.is_set() or done >= todo:  # freeze the projection at the end of extraction
                t_extract = t_extract or elapsed
                elapsed = t_extract
            say("progress: " + progress_line(done, todo, elapsed, len(texts), FULL_ARTICLES, gauges()))
            if session_time_up(T0, time.time(), stop_after_h):
                aborted["reason"] = f"session time limit ({stop_after_h:.1f} h) — resume in a new session"
                say(f"WATCHDOG: {aborted['reason']} — stopping graphrag, cache kept")
                proc.terminate()
                return
            if not extract_done.is_set() and done < todo and over_budget(done, todo, elapsed, budget_s):
                aborted["reason"] = f"projected extraction time over {budget_s / 3600:.1f} h budget"
                say(f"WATCHDOG: {aborted['reason']} — stopping graphrag")
                proc.terminate()
                return

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    with (LOG / "graphrag_index.log").open("w") as f:
        for line in proc.stdout:
            f.write(line)
            if line.strip().startswith("Workflow complete"):
                if "extract_graph" in line:
                    extract_done.set()
                now, c = time.time(), counters()
                stages.append({"stage": line.strip()[-120:], "seconds": now - last,
                               **{k: c[k] - c_last[k] for k in c}})
                last, c_last = now, c
                say(f"stage done: {stages[-1]}")
    rc = proc.wait()
    stop.set()
    say(f"graphrag index rc={rc}")
    info["index_rc"] = rc
    info["aborted"] = aborted.get("reason")
    if aborted or rc != 0:  # one file for the next session's dne-graphrag-cache dataset
        say(f"packed cache → {pack_cache(root, OUT / 'cache.tar.gz')}")
        if (root / "output" / "entities.parquet").exists():
            say(f"packed output tables → {pack_output(root, OUT / 'output.tar.gz')}")
    info["n_articles_indexed"] = len(texts)
    info["stages"] = stages
    with (LOG / "index_stages.csv").open("w") as f:
        f.write("stage,seconds,prompt_tokens,generation_tokens,requests\n")
        for s in stages:
            f.write(f"\"{s['stage']}\",{s['seconds']:.1f},{s['prompt_tokens']:.0f},"
                    f"{s['generation_tokens']:.0f},{s['requests']:.0f}\n")

# 4. totals
wall = time.time() - T0
info.update(wall_seconds=wall, gpu_hours=wall * len(info["gpus"]) / 3600,
            kwh=energy_kwh(LOG / "power.csv"), totals=counters())
(LOG / "run_info.json").write_text(json.dumps(info, indent=2))
say(f"done: {json.dumps({k: info[k] for k in ('wall_seconds', 'gpu_hours', 'kwh')})}")
