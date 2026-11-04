"""Mocked Gemini calls only; no credentials or live API requests."""

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock, patch
import json
import traceback

import pytest
from google.genai import errors, types

from gemini_models import GeminiModels, ModelCallError, ModelResponseError
from instrumentation import InstrumentedGenerator
from model_config import ModelConfig, load_model_config


RATES = {"input_usd_per_million": "0.25", "output_usd_per_million": "1.50"}
CONFIG = ModelConfig()
SYNTHETIC_SECRET = "synthetic-sensitive-value"


def response(text="Synthetic answer", usage=True):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(
            role="model", parts=[types.Part.from_text(text=text)],
        ))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=100, candidates_token_count=20,
            thoughts_token_count=5, total_token_count=125,
        ) if usage else None,
    )


def setup_layer(sdk_response=None, *, pricing=None, recorder=None, config=CONFIG):
    client = Mock()
    client.models.generate_content.return_value = response() if sdk_response is None else sdk_response
    records = []
    generator = InstrumentedGenerator(
        client, {config.small_model: RATES, config.large_model: RATES} if pricing is None else pricing,
        records.append if recorder is None else recorder,
    )
    return GeminiModels(generator, config=config), client, records


@pytest.mark.parametrize("role,model", [
    ("small", CONFIG.small_model), ("large", CONFIG.large_model),
])
def test_each_role_uses_same_interface_and_one_instrumented_call(role, model):
    layer, client, records = setup_layer()
    options = types.GenerateContentConfig(max_output_tokens=100)
    with patch("instrumentation.time.perf_counter", side_effect=[1, 1.125]):
        result = layer.generate("Synthetic query", role=role, generation_config=options)
    client.models.generate_content.assert_called_once_with(
        model=model, contents="Synthetic query", config=options,
    )
    assert result.text == "Synthetic answer" and result.model == model and result.role == role
    assert result.input_tokens == 100 and result.output_tokens == 20
    assert result.thinking_tokens == 5 and result.billed_output_tokens == 25
    assert result.latency_ms == 125
    assert Decimal(result.estimated_cost_usd) == Decimal("0.0000625")
    assert result.cost_status == "estimated" and result.timestamp_utc.endswith("+00:00")
    assert len(records) == 1
    assert records[0]["request_kind"] == f"{role}_model"
    assert records[0]["estimated_cost_usd"] == result.estimated_cost_usd
    assert "Synthetic query" not in json.dumps(records)
    client.models.embed_content.assert_not_called()


def test_configuration_defaults():
    assert load_model_config({}) == ModelConfig("gemini-3.1-flash-lite", "gemini-3.8-flash")


@pytest.mark.parametrize("variable,field", [
    ("GEMINI_SMALL_MODEL", "small_model"), ("GEMINI_LARGE_MODEL", "large_model"),
])
def test_environment_overrides_are_independent(monkeypatch, variable, field):
    # Replace the mapping, so only synthetic configuration is accessed.
    monkeypatch.setattr("model_config.os.environ", {variable: "gemini-custom"})
    config = load_model_config()
    assert getattr(config, field) == "gemini-custom"
    other = "large_model" if field == "small_model" else "small_model"
    assert getattr(config, other) == getattr(CONFIG, other)
    layer, client, _ = setup_layer(config=config)
    layer.generate("Question", role="small" if field == "small_model" else "large")
    assert client.models.generate_content.call_args.kwargs["model"] == "gemini-custom"


def test_config_loader_reads_only_model_names():
    class OnlyModelNames:
        def get(self, key, default):
            assert key in ("GEMINI_SMALL_MODEL", "GEMINI_LARGE_MODEL")
            return default
    assert load_model_config(OnlyModelNames()) == CONFIG


@pytest.mark.parametrize("invalid", ["", " ", None, 1, True, "bad\nmodel", "https://example.test/key"])
@pytest.mark.parametrize("field", ["small_model", "large_model"])
def test_invalid_model_names_fail_without_echoing_values(field, invalid):
    with pytest.raises(ValueError, match=field):
        ModelConfig(**{field: invalid})


def test_bad_environment_value_not_exposed():
    with pytest.raises(ValueError) as error:
        load_model_config({"GEMINI_SMALL_MODEL": SYNTHETIC_SECRET})
    assert SYNTHETIC_SECRET not in str(error.value)


@pytest.mark.parametrize("role", ["small", "large"])
def test_missing_usage_is_unavailable_not_zero(role):
    layer, _, records = setup_layer(response(usage=False))
    result = layer.generate("Question", role=role)
    assert result.input_tokens is result.output_tokens is result.estimated_cost_usd is None
    assert result.cost_status == "usage_unavailable"
    assert result.latency_ms is not None and len(records) == 1


def test_zero_usage_preserved():
    sdk_response = response()
    sdk_response.usage_metadata = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=0, candidates_token_count=0, thoughts_token_count=0, total_token_count=0,
    )
    layer, _, _ = setup_layer(sdk_response)
    result = layer.generate("Question", role="small")
    assert result.input_tokens == result.output_tokens == 0
    assert Decimal(result.estimated_cost_usd) == 0


def test_unknown_pricing_does_not_borrow_other_models_rate():
    layer, _, _ = setup_layer(pricing={CONFIG.small_model: RATES})
    result = layer.generate("Question", role="large")
    assert result.input_tokens == 100
    assert result.estimated_cost_usd is None and result.cost_status == "pricing_unavailable"


@pytest.mark.parametrize("sdk_response", [
    None, SimpleNamespace(), types.GenerateContentResponse(),
    response(""), response("   "),
    types.GenerateContentResponse(candidates=[types.Candidate(finish_reason="SAFETY")]),
])
def test_missing_or_empty_response_text_fails_safely(sdk_response):
    layer, client, records = setup_layer()
    client.models.generate_content.return_value = sdk_response
    with pytest.raises(ModelResponseError, match="no usable text"):
        layer.generate("Question", role="small")
    assert len(records) == 1 and records[0]["status"] == "success"


def test_parts_are_combined_and_thoughts_are_excluded():
    sdk_response = response()
    sdk_response.candidates[0].content.parts = [
        types.Part(text="private thought", thought=True),
        types.Part(text="  Final "), types.Part(text="answer  "),
    ]
    layer, _, _ = setup_layer(sdk_response)
    assert layer.generate("Question", role="small").text == "Final answer"


@pytest.mark.parametrize("role", ["small", "large"])
def test_api_errors_are_recorded_once_and_secret_safe(role, capsys):
    layer, client, records = setup_layer()
    client.models.generate_content.side_effect = errors.ClientError(
        403, {"error": {"message": SYNTHETIC_SECRET}},
    )
    with pytest.raises(ModelCallError) as caught:
        layer.generate("private query", role=role)
    rendered = "".join(traceback.format_exception(caught.value))
    output = capsys.readouterr()
    assert SYNTHETIC_SECRET not in rendered + output.out + output.err + json.dumps(records)
    assert caught.value.__suppress_context__
    assert len(records) == 1 and records[0]["status"] == "error"
    assert records[0]["estimated_cost_usd"] is None
    assert client.models.generate_content.call_count == 1


def test_response_parsing_errors_are_secret_safe(capsys):
    class BrokenResponse:
        @property
        def candidates(self):
            raise RuntimeError(SYNTHETIC_SECRET)
    layer, _, _ = setup_layer(BrokenResponse())
    with pytest.raises(ModelResponseError) as error:
        layer.generate("Question", role="small")
    assert SYNTHETIC_SECRET not in "".join(traceback.format_exception(error.value))
    assert SYNTHETIC_SECRET not in capsys.readouterr().err


def test_recorder_failure_preserves_returned_metrics(capsys):
    layer, _, _ = setup_layer(recorder=Mock(side_effect=OSError(SYNTHETIC_SECRET)))
    result = layer.generate("Question", role="small")
    assert result.text and result.input_tokens == 100
    assert result.cost_status == "estimated"
    assert SYNTHETIC_SECRET not in capsys.readouterr().err


def test_metric_extraction_failure_preserves_answer(capsys):
    layer, _, _ = setup_layer()
    with patch("instrumentation.usage_metrics", side_effect=ValueError(SYNTHETIC_SECRET)):
        result = layer.generate("Question", role="small")
    assert result.text == "Synthetic answer" and result.cost_status == "metrics_unavailable"
    assert result.estimated_cost_usd is None
    assert SYNTHETIC_SECRET not in capsys.readouterr().err


def test_snapshot_is_independent_of_recorder_mutation_and_later_calls():
    def mutate(record):
        record["input_tokens"] = 999
    layer, _, _ = setup_layer(recorder=mutate)
    first = layer.generate("Question", role="small")
    second = layer.generate("Question", role="large")
    assert first.input_tokens == second.input_tokens == 100
    assert first.model == CONFIG.small_model and second.model == CONFIG.large_model


@pytest.mark.parametrize("query", ["", " ", None, 2, []])
def test_invalid_query_never_calls_provider(query):
    layer, client, records = setup_layer()
    with pytest.raises(ValueError, match="Query"):
        layer.generate(query, role="small")
    client.models.generate_content.assert_not_called()
    assert not records


@pytest.mark.parametrize("role", ["medium", None, 2, [], "automatic"])
def test_invalid_role_never_calls_provider(role):
    layer, client, _ = setup_layer()
    with pytest.raises(ValueError, match="role"):
        layer.generate("Question", role=role)
    client.models.generate_content.assert_not_called()


def test_injected_configuration_and_generator_are_validated():
    with pytest.raises(ValueError, match="generator"):
        GeminiModels(object(), config=CONFIG)
    layer, _, _ = setup_layer()
    with pytest.raises(ValueError, match="config"):
        GeminiModels(layer.generator, config={})


def test_layer_can_load_model_configuration_without_reading_credentials(monkeypatch):
    layer, client, _ = setup_layer()
    monkeypatch.setattr("model_config.os.environ", {"GEMINI_LARGE_MODEL": "gemini-custom"})
    other = GeminiModels(layer.generator)
    other.generate("Question", role="large")
    assert client.models.generate_content.call_args.kwargs["model"] == "gemini-custom"
