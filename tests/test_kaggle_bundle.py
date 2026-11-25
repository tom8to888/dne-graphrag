import json

import yaml

from dne.kaggle_bundle import build_code_bundle


def test_build_code_bundle(tmp_path):
    out = build_code_bundle(tmp_path / "code", mode="index", n_articles=20,
                            llm_model="Qwen/Qwen2.5-14B-Instruct-AWQ", embed_model="BAAI/bge-m3",
                            username="tester")
    cfg = json.loads((out / "run_config.json").read_text())
    assert cfg == {"mode": "index", "n_articles": 20, "prompt_variant": "default", "llm_model": "Qwen/Qwen2.5-14B-Instruct-AWQ",
                   "embed_model": "BAAI/bge-m3", "graphrag": "3.3.0", "vllm": "0.33.0", "budget_hours": 9.0,
                   "stop_after_hours": 11.0, "resume": False, "restore_output": False, "aids": None}
    assert (out / "dne/vllm_metrics.py").exists() and (out / "dne/power.py").exists()
    assert (out / "dne/progress.py").exists()  # the watchdog
    assert (out / "dne/resume.py").exists()
    assert (out / "serve.sh").exists()
    assert (out / "prompts/extract_graph.txt").exists()
    settings = yaml.safe_load((out / "settings.yaml").read_text())
    assert settings["completion_models"]["default_completion_model"]["model"] == "Qwen/Qwen2.5-14B-Instruct-AWQ"
    meta = json.loads((out / "dataset-metadata.json").read_text())
    assert meta["id"] == "tester/dne-graphrag-code" and meta["isPrivate"] is True


def test_prompt_variant_domain(tmp_path):
    out = build_code_bundle(tmp_path / "code", mode="index", n_articles=50, llm_model="m", embed_model="e",
                            username="tester", prompt_variant="domain")
    from dne.paths import ROOT
    assert (out / "prompts/extract_graph.txt").read_text() == \
        (ROOT / "kaggle/common/prompts_domain/extract_graph.txt").read_text()
    assert json.loads((out / "run_config.json").read_text())["prompt_variant"] == "domain"


def test_budget_hours_in_run_config(tmp_path):
    out = build_code_bundle(tmp_path / "code", mode="index", n_articles=10, llm_model="m", embed_model="e",
                            username="tester", budget_hours=0.5)
    assert json.loads((out / "run_config.json").read_text())["budget_hours"] == 0.5


def test_subset_and_resume_in_run_config(tmp_path):
    out = build_code_bundle(tmp_path / "code", mode="index", n_articles=3, llm_model="m", embed_model="e",
                            username="tester", aids=[5, 2, 9], resume=True, stop_after_hours=10.5)
    cfg = json.loads((out / "run_config.json").read_text())
    assert cfg["aids"] == [2, 5, 9] and cfg["resume"] is True and cfg["stop_after_hours"] == 10.5


def test_prompt_variant_hybrid(tmp_path):
    # graphrag's own extraction prompt, but our tennis entity types
    out = build_code_bundle(tmp_path / "code", mode="index", n_articles=10, llm_model="m", embed_model="e",
                            username="tester", prompt_variant="hybrid")
    from dne.graphrag_config import ENTITY_TYPES
    from dne.paths import ROOT
    assert (out / "prompts/extract_graph.txt").read_text() == \
        (ROOT / "kaggle/common/prompts/extract_graph.txt").read_text()
    assert yaml.safe_load((out / "settings.yaml").read_text())["extract_graph"]["entity_types"] == ENTITY_TYPES


def test_finish_mode_in_run_config(tmp_path):
    out = build_code_bundle(tmp_path / "code", mode="index", n_articles=0, llm_model="m", embed_model="e",
                            username="u", restore_output=True,
                            workflows=["create_community_reports", "generate_text_embeddings"])
    assert json.loads((out / "run_config.json").read_text())["restore_output"] is True
    assert yaml.safe_load((out / "settings.yaml").read_text())["workflows"][0] == "create_community_reports"


def test_query_mode_ships_questions_and_worker(tmp_path):
    out = build_code_bundle(tmp_path / "code", mode="query", n_articles=0, llm_model="m", embed_model="e",
                            username="u", per_kind=2, budget_hours=1.0)
    cfg = json.loads((out / "run_config.json").read_text())
    assert cfg["mode"] == "query" and cfg["per_kind"] == 2
    assert cfg["systems"] == ["basic", "local", "global"]
    assert cfg["query_concurrency"] == 3 and cfg["global_level"] == 0  # C0, 3 at once
    assert cfg["query_stop_hours"] is None  # no hard stop unless asked (quota-limited sessions set one)
    assert (out / "questions.csv").exists()
    for f in ("query_worker.py", "query_plan.py", "retrieval.py"):
        assert (out / "dne" / f).exists()


def test_index_mode_has_no_query_keys(tmp_path):
    out = build_code_bundle(tmp_path / "code", mode="index", n_articles=1, llm_model="m", embed_model="e",
                            username="u")
    cfg = json.loads((out / "run_config.json").read_text())
    assert "per_kind" not in cfg and not (out / "questions.csv").exists()


def test_query_kernel_runs_same_script_as_index():
    from dne.paths import ROOT
    assert (ROOT / "kaggle/query/run_index.py").read_text() == (ROOT / "kaggle/index/run_index.py").read_text()


def test_query_timeout_in_run_config(tmp_path):
    out = build_code_bundle(tmp_path / "code", mode="query", n_articles=0, llm_model="m", embed_model="e",
                            username="u", query_stop_hours=4.5, query_timeout_s=1500.0)
    cfg = json.loads((out / "run_config.json").read_text())
    assert cfg["query_stop_hours"] == 4.5 and cfg["query_timeout_s"] == 1500.0


def test_judge_mode_ships_answers_and_judge_code(tmp_path):
    out = build_code_bundle(tmp_path / "code", mode="judge", n_articles=0, llm_model="m", embed_model="none",
                            username="u", judge_limit=5, judge_concurrency=6, query_stop_hours=1.0)
    cfg = json.loads((out / "run_config.json").read_text())
    assert cfg["mode"] == "judge" and cfg["judge_limit"] == 5 and cfg["judge_concurrency"] == 6
    assert cfg["query_stop_hours"] == 1.0 and cfg["embed_model"] == "none"
    assert (out / "answers.parquet").exists() and (out / "questions.csv").exists()
    for f in ("local_judge.py", "judge.py", "spend.py"):
        assert (out / "dne" / f).exists()


def test_query_8k_global_only_fresh(tmp_path):
    # global search alone at the paper's 8k context, not resuming the 12k answers
    out = build_code_bundle(tmp_path / "code", mode="query", n_articles=0, llm_model="m", embed_model="e",
                            username="u", systems=["global"], resume_answers=False, max_context_tokens=8000)
    cfg = json.loads((out / "run_config.json").read_text())
    assert cfg["systems"] == ["global"] and cfg["resume_answers"] is False
    assert cfg["max_context_tokens"] == 8000
    s = yaml.safe_load((out / "settings.yaml").read_text())
    assert s["global_search"]["max_context_tokens"] == 8000 and s["global_search"]["data_max_tokens"] == 8000


def test_query_defaults_resume_all_systems_12k(tmp_path):
    out = build_code_bundle(tmp_path / "code", mode="query", n_articles=0, llm_model="m", embed_model="e",
                            username="u")
    cfg = json.loads((out / "run_config.json").read_text())
    assert cfg["systems"] == ["basic", "local", "global"] and cfg["resume_answers"] is True
    assert cfg["max_context_tokens"] == 12000


def test_query_rejects_unknown_system(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        build_code_bundle(tmp_path / "code", mode="query", n_articles=0, llm_model="m", embed_model="e",
                          username="u", systems=["globl"])


def test_fresh_run_can_resume_from_its_own_answers(tmp_path):
    # a stopped 8k run continues from its own partial answers (shipped in the code bundle), never the 12k ones
    part = tmp_path / "answers_8k_partial.jsonl"
    part.write_text('{"qid": "q1", "system": "basic", "error": ""}\n')
    out = build_code_bundle(tmp_path / "code", mode="query", n_articles=0, llm_model="m", embed_model="e",
                            username="u", systems=["basic", "global"], resume_answers=False,
                            resume_file=part, max_context_tokens=8000)
    assert (out / "answers_resume.jsonl").read_text() == part.read_text()
    assert json.loads((out / "run_config.json").read_text())["resume_file"] is True


def test_chat_mode_ships_dialogue_and_chat_worker(tmp_path):
    d = tmp_path / "dialogue.txt"
    d.write_text("My juniors lose focus.\nWhich drills help?\n")
    out = build_code_bundle(tmp_path / "code", mode="chat", n_articles=0, llm_model="m", embed_model="e",
                            username="u", dialogue=d, query_stop_hours=1.0)
    cfg = json.loads((out / "run_config.json").read_text())
    assert cfg["mode"] == "chat" and cfg["systems"] == ["basic", "local", "global"]
    assert cfg["global_level"] == 0 and cfg["max_context_tokens"] == 12000 and cfg["query_stop_hours"] == 1.0
    assert (out / "dialogue.txt").read_text() == d.read_text()
    for f in ("chat.py", "chat_worker.py", "query_worker.py"):
        assert (out / "dne" / f).exists()
