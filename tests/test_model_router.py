"""Offline router tests: real classifier/model wrapper with mocked SDK generation."""

from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
import json
import traceback
from unittest.mock import Mock, patch

import pytest
from google.genai import errors, types

import model_router
from complexity_config import ComplexityConfig
from gemini_models import GeminiModels, ModelResponse
from instrumentation import InstrumentedGenerator
from model_config import ModelConfig
from model_router import (
    ClassificationError, ModelRouter, RoutingConfig, RoutingGenerationError,
)
from query_complexity import classify_query


CONFIG = ModelConfig()
RATES = {"input_usd_per_million": "0.25", "output_usd_per_million": "1.50"}
SIMPLE = "What is the capital of France?"
COMPLEX = "Write a Python function to merge two sorted lists."
SECRET = "synthetic-secret-do-not-display"


def setup_router(*, pricing=None, config=RoutingConfig(), model_config=CONFIG):
    client = Mock()
    client.models.generate_content.return_value = types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(
            role="model", parts=[types.Part.from_text(text="Synthetic answer")],
        ))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=100, candidates_token_count=20,
            thoughts_token_count=5, total_token_count=125,
        ),
    )
    records = []
    generator = InstrumentedGenerator(
        client, {model_config.small_model: RATES} if pricing is None else pricing, records.append,
    )
    models = GeminiModels(generator, config=model_config)
    return ModelRouter(models, config=config), client, records


@pytest.mark.parametrize("query,role,model,label", [
    (SIMPLE, "small", CONFIG.small_model, "simple"),
    (COMPLEX, "large", CONFIG.large_model, "complex"),
])
def test_default_routes_and_one_call_through_each_layer(query, role, model, label):
    router, client, records = setup_router()
    options = types.GenerateContentConfig(max_output_tokens=100)
    with (
        patch("model_router.classify_query", wraps=classify_query) as classifier,
        patch.object(router.models, "generate", wraps=router.models.generate) as generate,
    ):
        result = router.generate(query, generation_config=options)
    classifier.assert_called_once_with(query, config=router.config.complexity)
    generate.assert_called_once_with(query, role=role, generation_config=options)
    client.models.generate_content.assert_called_once_with(
        model=model, contents=query, config=options,
    )
    assert result.classification.label == label
    assert result.decision.selected_role == role
    assert f"label {label}" in result.decision.reason
    assert f"score {result.classification.score}" in result.decision.reason
    assert result.response.model == model and result.response.role == role
    assert result.decision.policy == "complexity-v1"
    assert not result.decision.cost_optimized
    assert len(records) == 1
    assert records[0]["request_kind"] == f"{role}_model"
    client.models.embed_content.assert_not_called()


def test_classifier_label_is_respected_without_recomputing_score(monkeypatch):
    router, client, _ = setup_router()
    classification = replace(classify_query(SIMPLE), label="complex", score=0)
    classifier = Mock(return_value=classification)
    monkeypatch.setattr(model_router, "classify_query", classifier)
    result = router.generate(SIMPLE)
    assert result.classification is classification
    assert result.decision.selected_role == "large"
    assert client.models.generate_content.call_args.kwargs["model"] == CONFIG.large_model


def test_response_object_and_all_metadata_are_preserved(monkeypatch):
    router, _, _ = setup_router()
    response = ModelResponse(
        text="Answer", model=CONFIG.small_model, role="small",
        input_tokens=10, output_tokens=2, thinking_tokens=1, billed_output_tokens=3,
        latency_ms=123.5, estimated_cost_usd="0.000007", cost_status="estimated",
        timestamp_utc="2026-11-04T00:00:00+00:00",
    )
    generate = Mock(return_value=response)
    monkeypatch.setattr(router.models, "generate", generate)
    result = router.generate(SIMPLE)
    assert result.response is response
    assert generate.call_count == 1


def test_real_instrumentation_cost_latency_and_usage_survive_routing():
    router, _, records = setup_router()
    with patch("instrumentation.time.perf_counter", side_effect=[10, 10.25]):
        result = router.generate(SIMPLE)
    assert result.response.input_tokens == 100
    assert result.response.output_tokens == 20
    assert result.response.thinking_tokens == 5
    assert result.response.billed_output_tokens == 25
    assert result.response.latency_ms == 250
    assert Decimal(result.response.estimated_cost_usd) == Decimal("0.0000625")
    assert result.response.cost_status == "estimated"
    assert result.response.timestamp_utc == records[0]["timestamp_utc"]


@pytest.mark.parametrize("query", [SIMPLE, COMPLEX])
def test_unknown_pricing_stays_unknown(query):
    router, _, _ = setup_router(pricing={})
    result = router.generate(query)
    assert result.response.estimated_cost_usd is None
    assert result.response.cost_status == "pricing_unavailable"
    assert result.response.input_tokens == 100
    assert result.decision.pricing_status == "unavailable"
    assert "Pricing is unavailable" in result.decision.reason
    assert not result.decision.cost_optimized


def test_configured_pricing_does_not_imply_known_cost_without_usage():
    router, client, _ = setup_router()
    client.models.generate_content.return_value.usage_metadata = None
    result = router.generate(SIMPLE)
    assert result.decision.pricing_status == "configured"
    assert result.response.input_tokens is result.response.output_tokens is None
    assert result.response.estimated_cost_usd is None
    assert result.response.cost_status == "usage_unavailable"


def test_reported_zero_remains_zero():
    router, client, _ = setup_router()
    client.models.generate_content.return_value.usage_metadata = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=0, candidates_token_count=0, thoughts_token_count=0, total_token_count=0,
    )
    result = router.generate(SIMPLE)
    assert result.response.input_tokens == result.response.output_tokens == 0
    assert Decimal(result.response.estimated_cost_usd) == 0
    assert result.response.cost_status == "estimated"


def test_pricing_does_not_invent_a_cheapest_model_policy():
    router, client, _ = setup_router(pricing={
        CONFIG.small_model: {"input_usd_per_million": "100", "output_usd_per_million": "100"},
        CONFIG.large_model: {"input_usd_per_million": "0", "output_usd_per_million": "0"},
    })
    result = router.generate(SIMPLE)
    assert result.decision.selected_role == "small"
    assert not result.decision.cost_optimized
    client.models.generate_content.assert_called_once()


def test_role_mapping_and_classifier_settings_are_configurable():
    config = RoutingConfig(simple_role="large", complex_role="small")
    router, _, _ = setup_router(config=config)
    assert router.generate(SIMPLE).response.role == "large"
    assert router.generate(COMPLEX).response.role == "small"
    router, _, _ = setup_router(config=RoutingConfig(complexity=ComplexityConfig(complex_threshold=4)))
    assert router.generate(COMPLEX).response.role == "small"


def test_custom_model_names_are_used_without_changing_policy():
    custom = ModelConfig(small_model="gemini-custom-small", large_model="gemini-custom-large")
    router, client, _ = setup_router(model_config=custom)
    assert router.generate(SIMPLE).response.model == custom.small_model
    assert router.generate(COMPLEX).response.model == custom.large_model
    assert client.models.generate_content.call_count == 2


@pytest.mark.parametrize("query", [None, "", " ", 2, [], "...?!"])
def test_invalid_input_fails_before_generation(query):
    router, client, records = setup_router()
    with pytest.raises(ClassificationError):
        router.generate(query)
    client.models.generate_content.assert_not_called()
    assert records == []


def test_classification_exception_is_sanitized_and_never_calls_model(monkeypatch, capsys):
    router, client, records = setup_router()
    monkeypatch.setattr(model_router, "classify_query", Mock(side_effect=RuntimeError(SECRET)))
    with pytest.raises(ClassificationError) as caught:
        router.generate(SIMPLE)
    assert SECRET not in "".join(traceback.format_exception(caught.value))
    output = capsys.readouterr()
    assert not output.out and not output.err
    client.models.generate_content.assert_not_called()
    assert not records


@pytest.mark.parametrize("bad", [
    None, {"label": "simple"}, replace(classify_query(SIMPLE), label="medium"),
    replace(classify_query(SIMPLE), score=-1), replace(classify_query(SIMPLE), score=True),
])
def test_invalid_classifier_result_is_rejected(monkeypatch, bad):
    router, client, _ = setup_router()
    monkeypatch.setattr(model_router, "classify_query", Mock(return_value=bad))
    with pytest.raises(ClassificationError):
        router.generate(SIMPLE)
    client.models.generate_content.assert_not_called()


@pytest.mark.parametrize("query,role", [(SIMPLE, "small"), (COMPLEX, "large")])
def test_api_errors_have_no_retry_or_fallback(query, role, capsys):
    router, client, records = setup_router()
    client.models.generate_content.side_effect = errors.ServerError(
        503, {"error": {"message": SECRET}},
    )
    with pytest.raises(RoutingGenerationError) as caught:
        router.generate(query)
    assert caught.value.decision.selected_role == role
    assert SECRET not in "".join(traceback.format_exception(caught.value)) + json.dumps(records)
    output = capsys.readouterr()
    assert SECRET not in output.out + output.err
    assert client.models.generate_content.call_count == 1
    assert len(records) == 1 and records[0]["status"] == "error"


def test_empty_provider_response_does_not_trigger_another_model():
    router, client, records = setup_router()
    client.models.generate_content.return_value = None
    with pytest.raises(RoutingGenerationError):
        router.generate(SIMPLE)
    assert client.models.generate_content.call_count == len(records) == 1


def test_injected_generation_error_is_also_sanitized(monkeypatch):
    router, _, _ = setup_router()
    generate = Mock(side_effect=RuntimeError(SECRET))
    monkeypatch.setattr(router.models, "generate", generate)
    with pytest.raises(RoutingGenerationError) as caught:
        router.generate(SIMPLE)
    assert SECRET not in "".join(traceback.format_exception(caught.value))
    assert generate.call_count == 1


@pytest.mark.parametrize("field,value", [
    ("simple_role", "medium"), ("simple_role", None), ("simple_role", []),
    ("complex_role", "automatic"), ("complex_role", True), ("complexity", {}),
])
def test_invalid_routing_config(field, value):
    with pytest.raises(ValueError):
        RoutingConfig(**{field: value})


def test_invalid_constructor_configuration():
    router, _, _ = setup_router()
    with pytest.raises(ValueError):
        ModelRouter(object())
    with pytest.raises(ValueError):
        ModelRouter(router.models, config={})
    router.models.generator.pricing = None
    with pytest.raises(ValueError, match="pricing mapping"):
        ModelRouter(router.models)


def test_decisions_are_deterministic_and_results_immutable():
    router, _, _ = setup_router()
    first = router.generate(SIMPLE)
    second = router.generate(SIMPLE)
    assert first.classification == second.classification
    assert first.decision == second.decision
    with pytest.raises(FrozenInstanceError):
        first.decision.selected_role = "large"
    with pytest.raises(FrozenInstanceError):
        router.config.simple_role = "large"
