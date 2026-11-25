"""Render graphrag 3.3 settings.yaml for vLLM servers on the Kaggle box.

Fairness rule (spec §7): one generator model; the same final-answer context budget for basic,
local and global search; the same answer-length cap for every search mode (via `answer_model`,
the same LLM with max_tokens). `response_type` is a query-time argument in graphrag 3.3, so the
query runner passes it, not this file.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULT_SETTINGS = Path(__file__).resolve().parents[1] / "kaggle/common/settings.default.yaml"
ENTITY_TYPES = ["technique", "tactic", "drill", "injury", "physical_quality", "mental_skill",
                "equipment", "person", "organization", "event"]


def render_settings(llm_model: str, embed_model: str, *,
                    llm_base: str = "http://localhost:8000/v1",
                    embed_base: str = "http://localhost:8001/v1",
                    max_context_tokens: int = 12000,
                    max_answer_tokens: int = 1500,
                    max_index_tokens: int = 4000,
                    concurrent_requests: int = 25,
                    chunk_size: int | None = None,
                    chunk_overlap: int | None = None,
                    entity_types: list[str] | None = None,
                    workflows: list[str] | None = None) -> str:
    """workflows: run only these graphrag workflows (on tables already in output/), e.g. to finish a stopped index."""
    cfg = yaml.safe_load(DEFAULT_SETTINGS.read_text())

    chat = {"model_provider": "openai", "model": llm_model, "api_base": llm_base,
            "auth_method": "api_key", "api_key": "not-needed", "retry": {"type": "exponential_backoff"}}
    index = copy.deepcopy(chat) | {"call_args": {"max_tokens": max_index_tokens}}  # stops runaway loops
    answer = copy.deepcopy(chat) | {"call_args": {"max_tokens": max_answer_tokens}}
    cfg["completion_models"] = {"default_completion_model": index, "answer_model": answer}
    cfg["concurrent_requests"] = concurrent_requests
    cfg["embedding_models"] = {"default_embedding_model": {
        "model_provider": "openai", "model": embed_model, "api_base": embed_base,
        "auth_method": "api_key", "api_key": "not-needed", "retry": {"type": "exponential_backoff"}}}

    if chunk_size is not None:
        cfg["chunking"]["size"] = chunk_size
    if chunk_overlap is not None:
        cfg["chunking"]["overlap"] = chunk_overlap
    cfg["extract_graph"]["entity_types"] = entity_types or ENTITY_TYPES

    for mode in ("local_search", "basic_search", "global_search", "drift_search"):
        cfg[mode]["completion_model_id"] = "answer_model"
    for mode in ("local_search", "basic_search", "global_search"):
        cfg[mode]["max_context_tokens"] = max_context_tokens
    cfg["global_search"]["data_max_tokens"] = max_context_tokens
    if workflows:
        cfg["workflows"] = list(workflows)
    return yaml.safe_dump(cfg, sort_keys=False)
