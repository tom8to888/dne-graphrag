import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from kaggle_run import code_options  # noqa: E402


def test_finish_option_runs_only_reports_and_embeddings():
    o = code_options(["finish=1", "conc=6"])
    assert o["restore_output"] is True and o["resume"] is True
    assert o["workflows"] == ["create_community_reports", "generate_text_embeddings"]
    assert o["concurrent_requests"] == 6


def test_old_budget_form_still_works():
    assert code_options(["14"]) == {"budget_hours": 14.0}


def test_per_kind_option_for_query_benchmark():
    assert code_options(["per_kind=2", "budget=0.75"]) == {"per_kind": 2, "budget_hours": 0.75}


def test_query_concurrency_and_global_level_options():
    assert code_options(["qconc=4", "glevel=1"]) == {"query_concurrency": 4, "global_level": 1}


def test_query_stop_option():
    assert code_options(["qstop=1.0"]) == {"query_stop_hours": 1.0}


def test_code_options_query_timeout():
    from tools.kaggle_run import code_options
    assert code_options(["qtimeout=1500"]) == {"query_timeout_s": 1500.0}


def test_code_options_judge_mode():
    from tools.kaggle_run import code_options
    assert code_options(["jlimit=5", "jconc=6"]) == {"judge_limit": 5, "judge_concurrency": 6}


def test_code_options_8k_global_fresh():
    assert code_options(["ctx=8000", "systems=global", "fresh=1"]) == {
        "max_context_tokens": 8000, "systems": ["global"], "resume_answers": False}


def test_code_options_resume_file(tmp_path):
    assert code_options([f"resume_file={tmp_path}/a.jsonl"]) == {"resume_file": tmp_path / "a.jsonl"}
