"""Read token counters from vLLM's Prometheus /metrics text."""
from __future__ import annotations

_NAMES = {
    "vllm:prompt_tokens_total": "prompt_tokens",
    "vllm:generation_tokens_total": "generation_tokens",
    "vllm:request_success_total": "requests",
}


def parse_token_counters(metrics_text: str) -> dict[str, float]:
    out = {v: 0.0 for v in _NAMES.values()}
    for line in metrics_text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in _NAMES:
            out[_NAMES[name]] += float(line.rsplit(" ", 1)[1])
    return out


_GAUGES = {
    "vllm:num_requests_running": "running",
    "vllm:num_requests_waiting": "waiting",
    "vllm:kv_cache_usage_perc": "kv_cache",
    "vllm:gpu_cache_usage_perc": "kv_cache",  # older name
}


def parse_gauges(metrics_text: str) -> dict[str, float]:
    """Requests running/waiting and KV-cache use (0–1), summed over engines."""
    out = {"running": 0.0, "waiting": 0.0, "kv_cache": 0.0}
    for line in metrics_text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in _GAUGES:
            out[_GAUGES[name]] += float(line.rsplit(" ", 1)[1])
    return out
