import pandas as pd
import pytest

from dne.query_plan import plan_jobs, projection_line, sources_to_aids

QS = pd.DataFrame({
    "id": ["L1", "L2", "L3", "C1", "C2", "B1"],
    "kind": ["lookup", "lookup", "lookup", "connecting", "connecting", "big_picture"],
    "split": ["test", "dev", "test", "dev", "test", "dev"],
})


def test_plan_all_question_major():
    jobs = plan_jobs(QS, ["basic", "local"], done=set())
    assert len(jobs) == 12
    assert jobs[:2] == [("L1", "basic"), ("L1", "local")]


def test_plan_skips_done():
    jobs = plan_jobs(QS, ["basic", "local"], done={("L1", "basic")})
    assert ("L1", "basic") not in jobs and len(jobs) == 11


def test_plan_benchmark_per_kind_prefers_dev():
    jobs = plan_jobs(QS, ["basic"], done=set(), per_kind=1)
    assert [q for q, _ in jobs] == ["L2", "C1", "B1"]


def test_plan_benchmark_fills_from_test_when_dev_short():
    jobs = plan_jobs(QS, ["basic"], done=set(), per_kind=2)
    assert sorted(q for q, _ in jobs) == ["B1", "C1", "C2", "L1", "L2"]


TU = pd.DataFrame({"id": ["u1", "u2", "u3"], "human_readable_id": [0, 1, 2], "document_id": ["d1", "d2", "d1"]})
DOCS = pd.DataFrame({"id": ["d1", "d2"], "title": ["138.txt", "62.txt"]})


def test_sources_short_ids_map_to_aids_in_order():
    ctx = {"sources": pd.DataFrame({"id": ["1", "0", "2"], "text": ["", "", ""]})}
    assert sources_to_aids(ctx, TU, DOCS) == [62, 138]


def test_sources_missing_gives_empty():
    assert sources_to_aids({"reports": pd.DataFrame({"id": ["5"]})}, TU, DOCS) == []
    assert sources_to_aids("not a dict", TU, DOCS) == []


def test_projection_line_scales_to_full_set():
    # 6 questions done in 600 s → 60 questions ≈ 6000 s = 100 min
    line = projection_line(n_done_q=6, elapsed=600, n_total_q=60, per_system={"basic": 100, "global": 400})
    assert "100 min" in line and "global" in line


def test_projection_line_needs_progress():
    with pytest.raises(ValueError):
        projection_line(n_done_q=0, elapsed=10, n_total_q=60, per_system={})


def test_system_broken_after_three_errors_in_a_row():
    from dne.query_plan import broken_systems
    rows = [{"system": "basic", "error": "empty answer"}] * 3 + [{"system": "local", "error": ""}]
    assert broken_systems(rows) == ["basic"]


def test_system_not_broken_if_any_success_among_first_three():
    from dne.query_plan import broken_systems
    rows = [{"system": "basic", "error": "x"}, {"system": "basic", "error": ""}, {"system": "basic", "error": "x"},
            {"system": "basic", "error": "x"}]
    assert broken_systems(rows) == []
