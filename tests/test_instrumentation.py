"""Offline tests: only mocked SDK calls, temporary files, and synthetic content."""

import json
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from google.genai import types

from instrumentation import InstrumentedGenerator, JsonlRecorder, load_pricing, usage_metrics


RATES = {
    "input_usd_per_million": "0.25",
    "output_usd_per_million": "1.50",
    "cached_input_usd_per_million": "0.025",
    "pricing_version": "test",
}


def response(**usage):
    usage.setdefault("cached_content_token_count", 0)
    usage.setdefault("tool_use_prompt_token_count", 0)
    return types.GenerateContentResponse(
        usage_metadata=types.GenerateContentResponseUsageMetadata(**usage)
    )


def test_tokens_thinking_and_cost():
    result = usage_metrics(response(
        prompt_token_count=1000, candidates_token_count=100,
        thoughts_token_count=50, total_token_count=1150,
    ), RATES)
    assert result["input_tokens"] == 1000
    assert result["output_tokens"] == 100
    assert result["thinking_tokens"] == 50
    assert result["billed_output_tokens"] == 150
    assert Decimal(result["estimated_cost_usd"]) == Decimal("0.000475")


@pytest.mark.parametrize("sdk_response", [
    SimpleNamespace(), types.GenerateContentResponse(),
    response(prompt_token_count=10),
    response(prompt_token_count=10, candidates_token_count=2),
])
def test_missing_usage_is_not_zero(sdk_response):
    result = usage_metrics(sdk_response, RATES)
    assert result["estimated_cost_usd"] is None
    assert result["cost_status"] == "usage_unavailable"


def test_explicit_zero_is_preserved():
    result = usage_metrics(response(
        prompt_token_count=0, candidates_token_count=0,
        thoughts_token_count=0, total_token_count=0,
    ), RATES)
    assert result["input_tokens"] == result["output_tokens"] == 0
    assert Decimal(result["estimated_cost_usd"]) == 0


def test_missing_thinking_can_use_total_without_overwriting_raw_metadata():
    result = usage_metrics(response(
        prompt_token_count=10, candidates_token_count=2, total_token_count=15,
    ), RATES)
    assert result["thinking_tokens"] is None
    assert result["billed_output_tokens"] == 5


def test_cached_input_discount():
    result = usage_metrics(response(
        prompt_token_count=1000, candidates_token_count=100,
        thoughts_token_count=0, cached_content_token_count=500,
    ), RATES)
    assert Decimal(result["estimated_cost_usd"]) == Decimal("0.0002875")


def test_unknown_model_pricing_and_unsupported_usage():
    sdk_response = response(prompt_token_count=10, candidates_token_count=2, thoughts_token_count=0)
    assert usage_metrics(sdk_response, None)["cost_status"] == "pricing_unavailable"
    for extra in [
        {"total_token_count": 1},
        {"tool_use_prompt_token_count": 5},
        {"cached_content_token_count": 20},
    ]:
        changed = response(prompt_token_count=10, candidates_token_count=2, thoughts_token_count=0, **extra)
        assert usage_metrics(changed, RATES)["estimated_cost_usd"] is None


def test_custom_pricing_and_validation(tmp_path):
    path = tmp_path / "pricing.json"
    path.write_text(json.dumps({"custom": {**RATES, "output_usd_per_million": "3"}}))
    pricing = load_pricing(path)
    result = usage_metrics(response(
        prompt_token_count=0, candidates_token_count=1, thoughts_token_count=0,
    ), pricing["custom"])
    assert Decimal(result["estimated_cost_usd"]) == Decimal("0.000003")
    path.write_text(json.dumps({"bad": {**RATES, "input_usd_per_million": "-1"}}))
    with pytest.raises(ValueError):
        load_pricing(path)


def test_wrapper_records_latency_returns_same_response_and_passes_arguments():
    client = Mock()
    sdk_response = response(prompt_token_count=10, candidates_token_count=2, total_token_count=12)
    client.models.generate_content.return_value = sdk_response
    records = []
    generator = InstrumentedGenerator(client, {"test-model": RATES}, records.append)
    with patch("instrumentation.time.perf_counter", side_effect=[5, 5.25]):
        assert generator.generate_content(model="test-model", request_kind="answer", contents="private") is sdk_response
    client.models.generate_content.assert_called_once_with(model="test-model", contents="private")
    assert records[0]["latency_ms"] == 250
    assert records[0]["timestamp_utc"].endswith("+00:00")
    assert records[0]["status"] == "success"
    assert "private" not in json.dumps(records)


def test_failed_attempts_recorded_separately_and_original_exception_preserved():
    client = Mock()
    error = RuntimeError("sensitive error details")
    client.models.generate_content.side_effect = error
    records = []
    generator = InstrumentedGenerator(client, {}, records.append)
    for _ in range(2):
        with pytest.raises(RuntimeError) as caught:
            generator.generate_content(model="test", request_kind="answer", contents="private")
        assert caught.value is error
    assert len(records) == 2
    assert all(r["status"] == "error" and r["latency_ms"] >= 0 for r in records)
    assert all(r["input_tokens"] is None and r["estimated_cost_usd"] is None for r in records)
    assert "sensitive" not in json.dumps(records)


@pytest.mark.parametrize("provider_fails", [False, True])
def test_recorder_failure_does_not_change_provider_result(provider_fails, capsys):
    client = Mock()
    original = RuntimeError("provider")
    if provider_fails:
        client.models.generate_content.side_effect = original
    recorder = Mock(side_effect=OSError("private path"))
    generator = InstrumentedGenerator(client, {}, recorder)
    if provider_fails:
        with pytest.raises(RuntimeError) as caught:
            generator.generate_content(model="test", request_kind="memory_summary")
        assert caught.value is original
    else:
        assert generator.generate_content(model="test", request_kind="answer") is client.models.generate_content.return_value
    assert capsys.readouterr().err == "Warning: request metrics could not be recorded.\n"


def test_jsonl_appends_and_preserves_nulls(tmp_path):
    path = tmp_path / "metrics.jsonl"
    recorder = JsonlRecorder(path)
    recorder({"input_tokens": None, "estimated_cost_usd": None})
    recorder({"input_tokens": 0, "estimated_cost_usd": "0"})
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records == [
        {"input_tokens": None, "estimated_cost_usd": None},
        {"input_tokens": 0, "estimated_cost_usd": "0"},
    ]


@pytest.mark.parametrize("payload", [[], {"test": {}}, {"test": None}])
def test_invalid_pricing_shape_is_rejected(tmp_path, payload):
    path = tmp_path / "pricing.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_pricing(path)
