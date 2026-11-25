import pytest

from dne.progress import eta_seconds, expected_extract_calls, over_budget, progress_line
from dne.vllm_metrics import parse_gauges


def test_expected_extract_calls_counts_chunks_and_gleanings():
    # 4.4 chars/token; 1200-token chunks with 100 overlap advance 1100 tokens
    texts = ["x" * int(4.4 * 2200), "y" * int(4.4 * 500)]
    assert expected_extract_calls(texts, chunk_size=1200, overlap=100, gleanings=1) == (2 + 1) * 2
    assert expected_extract_calls(texts, chunk_size=1200, overlap=100, gleanings=0) == 3


def test_eta_none_before_first_call():
    assert eta_seconds(0, 100, 60) is None


def test_eta_linear():
    assert eta_seconds(25, 100, 600) == pytest.approx(1800)


def test_over_budget_waits_for_grace():
    assert not over_budget(1, 100, 300, budget_s=3600, grace_s=900)  # too early to judge


def test_over_budget_after_grace():
    assert over_budget(10, 100, 1000, budget_s=3600, grace_s=900)  # 10% in 1000 s → 10,000 s total
    assert not over_budget(50, 100, 1000, budget_s=3600, grace_s=900)


def test_progress_line_projects_full_corpus():
    line = progress_line(done=26, total=262, elapsed=600, n_articles=50, full_articles=541,
                         gauges={"running": 25, "waiting": 3, "kv_cache": 0.97})
    assert "26/262" in line and "running 25" in line and "kv 97%" in line
    assert "full corpus ≈ 18.2 h" in line  # 600/26*262 s for 50 articles, ×541/50


GAUGES = """# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="m"} 25.0
vllm:num_requests_waiting{engine="0",model_name="m"} 3.0
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.97
vllm:prompt_tokens_total{engine="0",model_name="m"} 1500.0
"""


def test_parse_gauges():
    assert parse_gauges(GAUGES) == {"running": 25.0, "waiting": 3.0, "kv_cache": 0.97}


def test_parse_gauges_old_kv_name():
    assert parse_gauges('vllm:gpu_cache_usage_perc{engine="0"} 0.5\n')["kv_cache"] == 0.5


def test_stalled_after_no_new_results():
    from dne.progress import stalled
    assert not stalled(last_change=0, now=19 * 60, stall_s=20 * 60)
    assert stalled(last_change=0, now=20 * 60, stall_s=20 * 60)


# --- hang protection for the query run (21/11): one stuck query must not hold the GPU session open


def test_with_timeout_returns_the_result():
    import asyncio
    from dne.progress import with_timeout

    async def quick():
        return "answer"
    assert asyncio.run(with_timeout(quick(), 5)) == "answer"


def test_with_timeout_names_the_limit_when_a_query_hangs():
    import asyncio
    from dne.progress import with_timeout

    with pytest.raises(TimeoutError, match="over 0.05 s"):
        asyncio.run(with_timeout(asyncio.sleep(10), 0.05))


def test_worker_deadline_is_stop_plus_one_query_plus_margin():
    from dne.progress import worker_deadline_s
    assert worker_deadline_s(4.5, 1500) == 4.5 * 3600 + 1500 + 300
    assert worker_deadline_s(None, 1500) is None


def test_query_budget_does_not_judge_on_the_first_slow_answers():
    # 21/11 v4: 3 queries start together and finish only after ~8 min, so at 850 s (4 answers) the projection
    # read 8.2 h and a 5 h budget stopped the run. The query watchdog waits 30 min before it judges.
    from dne.progress import QUERY_BUDGET_GRACE_S, over_budget
    assert QUERY_BUDGET_GRACE_S >= 1800
    assert not over_budget(4, 139, 850, 8 * 3600, grace_s=QUERY_BUDGET_GRACE_S)
