from dne.vllm_metrics import parse_token_counters

SAMPLE = """# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{engine="0",model_name="m"} 1500.0
vllm:prompt_tokens_total{engine="1",model_name="m"} 500.0
vllm:generation_tokens_total{engine="0",model_name="m"} 300.0
vllm:request_success_total{engine="0",finished_reason="stop",model_name="m"} 4.0
vllm:request_success_total{engine="0",finished_reason="length",model_name="m"} 1.0
vllm:prompt_tokens_created{engine="0",model_name="m"} 1.7e9
"""


def test_parse_token_counters():
    assert parse_token_counters(SAMPLE) == {"prompt_tokens": 2000.0, "generation_tokens": 300.0, "requests": 5.0}


def test_parse_token_counters_empty():
    assert parse_token_counters("") == {"prompt_tokens": 0.0, "generation_tokens": 0.0, "requests": 0.0}
