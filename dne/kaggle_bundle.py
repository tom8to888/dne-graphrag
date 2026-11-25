"""Build the private Kaggle 'code' dataset the GPU kernels run from (Kaggle script kernels upload one file)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from dne.graphrag_config import render_settings
from dne.paths import ROOT

GRAPHRAG_VERSION = "3.3.0"
VLLM_VERSION = "0.33.0"
CODE_SLUG = "dne-graphrag-code"
_DNE_FILES = ("__init__.py", "vllm_metrics.py", "power.py", "progress.py", "resume.py",
              "retrieval.py", "query_plan.py", "query_worker.py", "chat.py", "chat_worker.py",
              "judge.py", "spend.py", "local_judge.py")
SYSTEMS = ["basic", "local", "global"]
PROMPT_VARIANTS = ("default", "domain", "hybrid", "auto")
GRAPHRAG_DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]


def build_code_bundle(out: Path, *, mode: str, n_articles: int, llm_model: str, embed_model: str,
                      username: str, prompt_variant: str = "default", budget_hours: float = 9.0,
                      stop_after_hours: float = 11.0, resume: bool = False, aids: list[int] | None = None,
                      restore_output: bool = False, per_kind: int | None = None,
                      query_concurrency: int = 3, global_level: int = 0,
                      query_stop_hours: float | None = None, query_timeout_s: float = 1500.0,
                      judge_limit: int | None = None, judge_concurrency: int = 6,
                      systems: list[str] | None = None, resume_answers: bool = True,
                      resume_file: Path | None = None, dialogue: Path | None = None, **settings_kw) -> Path:
    """prompt_variant: 'default' = graphrag's prompt + entity types; 'domain' = coaching prompt with few-shot
    examples + coaching entity types; 'hybrid' = graphrag's prompt + coaching entity types; 'auto' = default files, and the kernel runs `graphrag prompt-tune` first.
    budget_hours: the kernel's watchdog stops extraction when its projected time exceeds this.
    stop_after_hours: clean stop before Kaggle's 12 h limit, so the cache is saved for the next session.
    resume: restore cache.tar.gz from the dne-graphrag-cache dataset before indexing.
    aids: index exactly these articles (n_articles is then ignored).
    restore_output: also restore the stopped session's output tables (pair with workflows=[...] to finish it).
    mode='query': answer data/questions/questions.csv with basic/local/global search on the index dataset;
    per_kind = benchmark on that many questions per kind (dev first), budget_hours = the run's self-stop,
    query_concurrency = queries at once, global_level = community level for global search (0 = the paper's C0),
    query_timeout_s = one query is abandoned after this long (error row, retried by the next run),
    systems = answer only with these (default all three), resume_answers=False = ignore the earlier
    answers dataset (a fresh run, e.g. global at 8k context next to the 12k answers),
    resume_file = that fresh run's own partial answers.jsonl, shipped in the bundle so a rerun continues."""
    if prompt_variant not in PROMPT_VARIANTS:
        raise ValueError(prompt_variant)
    systems = list(systems or SYSTEMS)
    if not set(systems) <= set(SYSTEMS):
        raise ValueError(f"unknown system in {systems}")
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    (out / "dne").mkdir(parents=True)
    for name in _DNE_FILES:
        shutil.copy2(ROOT / "dne" / name, out / "dne" / name)
    common = ROOT / "kaggle/common"
    shutil.copy2(common / "serve.sh", out / "serve.sh")
    shutil.copytree(common / "prompts", out / "prompts")
    if prompt_variant == "domain":
        shutil.copy2(common / "prompts_domain/extract_graph.txt", out / "prompts/extract_graph.txt")
    elif prompt_variant != "hybrid":  # hybrid keeps render_settings' coaching ENTITY_TYPES
        settings_kw.setdefault("entity_types", GRAPHRAG_DEFAULT_ENTITY_TYPES)
    (out / "settings.yaml").write_text(render_settings(llm_model, embed_model, **settings_kw))
    run_config = {
        "mode": mode, "n_articles": n_articles, "prompt_variant": prompt_variant, "llm_model": llm_model, "embed_model": embed_model,
        "graphrag": GRAPHRAG_VERSION, "vllm": VLLM_VERSION,
        "budget_hours": budget_hours, "stop_after_hours": stop_after_hours, "resume": resume, "restore_output": restore_output,
        "aids": sorted(aids) if aids else None}
    if mode == "query":
        run_config.update(per_kind=per_kind, systems=systems, query_concurrency=query_concurrency,
                          global_level=global_level, query_stop_hours=query_stop_hours,
                          query_timeout_s=query_timeout_s, resume_answers=resume_answers,
                          max_context_tokens=settings_kw.get("max_context_tokens", 12000),
                          resume_file=resume_file is not None)
        if resume_file is not None:
            shutil.copy2(resume_file, out / "answers_resume.jsonl")
        shutil.copy2(ROOT / "data/questions/questions.csv", out / "questions.csv")
    if mode == "chat":  # scripted demo dialogue (brief: conversational agent), every system answers each turn
        if dialogue is None:
            raise ValueError("mode chat needs dialogue=<file, one user turn per line>")
        run_config.update(systems=systems, global_level=global_level, query_stop_hours=query_stop_hours,
                          query_timeout_s=query_timeout_s,
                          max_context_tokens=settings_kw.get("max_context_tokens", 12000))
        shutil.copy2(dialogue, out / "dialogue.txt")
    if mode == "judge":  # second judge on Kaggle: the finished answers in, judgments_local.jsonl out
        run_config.update(judge_limit=judge_limit, judge_concurrency=judge_concurrency,
                          query_stop_hours=query_stop_hours, query_timeout_s=query_timeout_s)
        shutil.copy2(ROOT / "data/questions/questions.csv", out / "questions.csv")
        shutil.copy2(ROOT / "results/answers.parquet", out / "answers.parquet")
    (out / "run_config.json").write_text(json.dumps(run_config, indent=2))
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": "DNE GraphRAG code", "id": f"{username}/{CODE_SLUG}",
        "licenses": [{"name": "other"}], "isPrivate": True}, indent=2))
    return out
