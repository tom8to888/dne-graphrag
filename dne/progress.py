"""Live progress, ETA and the over-budget watchdog for long Kaggle index runs.

Progress = graphrag's extract_graph cache entries (one per extraction or gleaning call) against
the expected number of calls, estimated from the input texts before graphrag starts.
"""
from __future__ import annotations

import math

QUERY_BUDGET_GRACE_S = 1800  # query run: the first answers (3 started together) take ~8 min — judge after 30 min (21/11)
CHARS_PER_TOKEN = 4.4  # measured on the 541 corpus articles (cl100k)


def expected_extract_calls(texts: list[str], chunk_size: int, overlap: int, gleanings: int) -> int:
    step = chunk_size - overlap
    chunks = sum(max(1, math.ceil(len(t) / CHARS_PER_TOKEN / step)) for t in texts)
    return chunks * (1 + gleanings)


def eta_seconds(done: int, total: int, elapsed: float) -> float | None:
    """Seconds still to go, assuming a constant rate; None until the first call has finished."""
    if done <= 0:
        return None
    return elapsed / done * max(total - done, 0)


def over_budget(done: int, total: int, elapsed: float, budget_s: float, grace_s: float = 900) -> bool:
    """True when, after the grace period, the projected stage time exceeds the budget."""
    if elapsed < grace_s:
        return False
    eta = eta_seconds(done, total, elapsed)
    return eta is None or elapsed + eta > budget_s


def progress_line(done: int, total: int, elapsed: float, n_articles: int, full_articles: int,
                  gauges: dict[str, float]) -> str:
    eta = eta_seconds(done, total, elapsed)
    parts = [f"extract {done}/{total}", f"{elapsed / 60:.0f} min"]
    if eta is not None:
        stage = elapsed + eta
        parts += [f"ETA {eta / 60:.0f} min",
                  f"full corpus ≈ {stage * full_articles / max(n_articles, 1) / 3600:.1f} h"]
    parts += [f"running {gauges.get('running', 0):.0f}", f"waiting {gauges.get('waiting', 0):.0f}",
              f"kv {gauges.get('kv_cache', 0) * 100:.0f}%"]
    return " | ".join(parts)


def stalled(last_change: float, now: float, stall_s: float = 1200) -> bool:
    """No new LLM result (any graphrag cache entry) for stall_s seconds: hung server or retry loop."""
    return now - last_change >= stall_s


async def with_timeout(coro, seconds: float):
    """Await `coro`, but give up after `seconds` with a TimeoutError that names the limit — one hung query
    becomes an error row (the next run retries it) instead of holding the GPU session open (21/11)."""
    import asyncio
    try:
        return await asyncio.wait_for(coro, seconds)
    except asyncio.TimeoutError:
        raise TimeoutError(f"query took over {seconds:g} s") from None


def worker_deadline_s(stop_after_hours: float | None, query_timeout_s: float, margin_s: float = 300) -> float | None:
    """When the kernel kills the query worker outright: no new queries start after `stop_after_hours`, the
    last one may take up to `query_timeout_s`, plus a margin. None when the run has no time limit."""
    if not stop_after_hours:
        return None
    return stop_after_hours * 3600 + query_timeout_s + margin_s
