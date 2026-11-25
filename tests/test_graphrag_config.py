import yaml

from dne.graphrag_config import render_settings


def _cfg(**kw):
    return yaml.safe_load(render_settings("Qwen/Qwen2.5-14B-Instruct-AWQ", "BAAI/bge-m3", **kw))


def test_render_settings_same_budget():
    c = _cfg(max_context_tokens=12000)
    assert c["local_search"]["max_context_tokens"] == 12000
    assert c["basic_search"]["max_context_tokens"] == 12000
    assert c["global_search"]["max_context_tokens"] == 12000
    assert c["global_search"]["data_max_tokens"] == 12000


def test_render_settings_points_local():
    c = _cfg()
    chat = c["completion_models"]["default_completion_model"]
    emb = c["embedding_models"]["default_embedding_model"]
    assert chat["api_base"] == "http://localhost:8000/v1"
    assert chat["model"] == "Qwen/Qwen2.5-14B-Instruct-AWQ"
    assert emb["api_base"] == "http://localhost:8001/v1"
    assert emb["model"] == "BAAI/bge-m3"


def test_answer_model_same_llm_with_length_cap():
    c = _cfg(max_answer_tokens=1500)
    ans = c["completion_models"]["answer_model"]
    assert ans["model"] == c["completion_models"]["default_completion_model"]["model"]
    assert ans["call_args"]["max_tokens"] == 1500
    for mode in ("local_search", "basic_search", "drift_search"):
        assert c[mode]["completion_model_id"] == "answer_model"
    assert c["global_search"]["completion_model_id"] == "answer_model"
    assert c["extract_graph"]["completion_model_id"] == "default_completion_model"


def test_indexing_model_capped():
    # pilot 15/11: uncapped extraction calls looped until the 16k context was full, timed out and
    # were retried, filling the KV cache for hours; p99 of real extraction outputs was 3.2k tokens
    c = _cfg()
    assert c["completion_models"]["default_completion_model"]["call_args"]["max_tokens"] == 4000
    assert _cfg(max_index_tokens=3000)["completion_models"]["default_completion_model"][
        "call_args"]["max_tokens"] == 3000


def test_concurrency_default_and_override():
    assert _cfg()["concurrent_requests"] == 25  # 48 overfilled the 69k-token KV cache (run v5)
    assert _cfg(concurrent_requests=16)["concurrent_requests"] == 16


def test_chunking_defaults_kept_unless_given():
    assert _cfg()["chunking"]["size"] == 1200
    assert _cfg(chunk_size=800, chunk_overlap=50)["chunking"] == {
        "type": "tokens", "size": 800, "overlap": 50, "encoding_model": "o200k_base"}


def test_workflows_only_when_given():
    assert "workflows" not in yaml.safe_load(render_settings("m", "e"))
    wf = ["create_community_reports", "generate_text_embeddings"]
    assert yaml.safe_load(render_settings("m", "e", workflows=wf))["workflows"] == wf
