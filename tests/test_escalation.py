"""Offline escalation tests using real components and synthetic SDK responses."""

from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
import socket
import traceback
from unittest.mock import Mock, patch

import pytest
from google.genai import types

import escalation
from answer_evaluator import evaluate_answer
from escalation import EscalationError, EscalationPipeline
from evaluation_config import EvaluationConfig
from gemini_models import GeminiModels
from instrumentation import InstrumentedGenerator
from model_config import ModelConfig
from model_router import ModelRouter, RoutingConfig
from query_complexity import classify_query


QUERY = "What is the capital of France?"
COMPLEX = "Write a Python function to merge two sorted lists."
CONFIG = ModelConfig()
RATES = {"input_usd_per_million": "0.25", "output_usd_per_million": "1.50"}
SECRET = "synthetic-secret-must-not-appear"


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Network access is forbidden in offline tests.")
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)


def sdk_response(text="Paris.", *, usage=True):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(
            role="model", parts=[types.Part.from_text(text=text)],
        ))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=100, candidates_token_count=20,
            thoughts_token_count=5, total_token_count=125,
        ) if usage else None,
    )


def setup_pipeline(*, replies=None, pricing=None, routing_config=RoutingConfig(),
                   model_config=CONFIG, evaluation_config=EvaluationConfig()):
    client = Mock()
    client.models.generate_content.side_effect = (
        [sdk_response()] if replies is None else replies
    )
    records = []
    generator = InstrumentedGenerator(
        client,
        {model_config.small_model: RATES, model_config.large_model: RATES}
        if pricing is None else pricing,
        records.append,
    )
    models = GeminiModels(generator, config=model_config)
    router = ModelRouter(models, config=routing_config)
    return EscalationPipeline(router, evaluation_config=evaluation_config), client, records


def quality(score, decision):
    return replace(evaluate_answer(QUERY, "Paris."), score=score, decision=decision)


def test_initial_pass_retains_original_objects_and_calls_each_component_once():
    pipeline, client, records = setup_pipeline()
    options = types.GenerateContentConfig(max_output_tokens=100)
    with (
        patch.object(pipeline.router, "generate", wraps=pipeline.router.generate) as route,
        patch("model_router.classify_query", wraps=classify_query) as classify,
        patch("escalation.evaluate_answer", wraps=evaluate_answer) as evaluate,
    ):
        result = pipeline.generate(QUERY, generation_config=options)
    route.assert_called_once_with(QUERY, generation_config=options)
    classify.assert_called_once()
    evaluate.assert_called_once_with(QUERY, "Paris.", config=pipeline.evaluation_config)
    client.models.generate_content.assert_called_once_with(
        model=CONFIG.small_model, contents=QUERY, config=options,
    )
    assert result.query == QUERY
    assert not result.escalated and result.escalation_role is None
    assert "passed" in result.escalation_reason
    assert result.total_attempts == 1 and result.final_attempt_index == 0
    assert result.initial_response is result.initial_routing.response
    assert result.final_response is result.initial_response
    assert result.final_evaluation is result.initial_evaluation
    assert result.final_evaluation.decision == "pass"
    assert result.escalation_response is result.escalation_evaluation is None
    assert result.initial_response.timestamp_utc == records[0]["timestamp_utc"]
    assert len(records) == 1
    client.models.embed_content.assert_not_called()


def test_small_fail_large_pass_with_real_evaluator_and_one_call_per_model():
    pipeline, client, records = setup_pipeline(replies=[
        sdk_response("I don't know the capital of France."),
        sdk_response("Paris."),
    ])
    options = types.GenerateContentConfig(max_output_tokens=100)
    with (
        patch("model_router.classify_query", wraps=classify_query) as classify,
        patch.object(pipeline.router.models, "generate", wraps=pipeline.router.models.generate) as generate,
        patch("escalation.evaluate_answer", wraps=evaluate_answer) as evaluate,
    ):
        result = pipeline.generate(QUERY, generation_config=options)
    classify.assert_called_once()
    assert result.escalated and result.total_attempts == 2
    assert result.escalation_role == "large"
    assert "failed" in result.escalation_reason and "once" in result.escalation_reason
    assert result.initial_evaluation.decision == "fail"
    assert result.escalation_evaluation.decision == "pass"
    # Passing wins even though uncertainty makes the initial score higher.
    assert result.initial_evaluation.score > result.final_evaluation.score
    assert result.final_response is result.escalation_response
    assert result.final_attempt_index == 1 and result.final_response.text == "Paris."
    assert result.initial_routing.decision.selected_role == "small"
    assert [call.kwargs["role"] for call in generate.call_args_list] == ["small", "large"]
    assert [call.kwargs for call in client.models.generate_content.call_args_list] == [
        {"model": CONFIG.small_model, "contents": QUERY, "config": options},
        {"model": CONFIG.large_model, "contents": QUERY, "config": options},
    ]
    assert evaluate.call_count == 2 and len(records) == 2
    assert all(call.kwargs["config"] is pipeline.evaluation_config for call in evaluate.call_args_list)


@pytest.mark.parametrize("first_score,second_score,selected", [
    (0.2, 0.7, 1), (0.7, 0.2, 0), (0.5, 0.5, 0), (0.0, 0.0, 0),
])
def test_both_fail_higher_score_wins_and_ties_keep_initial(first_score, second_score, selected):
    pipeline, client, _ = setup_pipeline(replies=[sdk_response("First"), sdk_response("Second")])
    evaluations = [quality(first_score, "fail"), quality(second_score, "fail")]
    with patch("escalation.evaluate_answer", side_effect=evaluations):
        result = pipeline.generate(QUERY)
    assert result.final_attempt_index == selected
    assert result.final_evaluation is evaluations[selected]
    assert result.final_response is result.attempts[selected].response
    assert result.final_evaluation.decision == "fail"
    assert client.models.generate_content.call_count == result.total_attempts == 2


def test_initial_large_failure_does_not_retry_or_fall_back():
    pipeline, client, _ = setup_pipeline(replies=[sdk_response("I don't know.")])
    result = pipeline.generate(COMPLEX)
    assert result.initial_routing.classification.label == "complex"
    assert result.initial_response.role == "large"
    assert result.final_evaluation.decision == "fail"
    assert not result.escalated and result.total_attempts == 1
    assert "already used" in result.escalation_reason
    client.models.generate_content.assert_called_once()


@pytest.mark.parametrize("query,config,expected_roles", [
    (QUERY, RoutingConfig(simple_role="large"), ["large"]),
    (COMPLEX, RoutingConfig(complex_role="small"), ["small", "large"]),
])
def test_actual_routing_roles_are_respected(query, config, expected_roles):
    pipeline, client, _ = setup_pipeline(
        routing_config=config,
        replies=[sdk_response("I don't know."), sdk_response("I don't know.")],
    )
    result = pipeline.generate(query)
    assert [attempt.role for attempt in result.attempts] == expected_roles
    assert client.models.generate_content.call_count == len(expected_roles)


def test_same_model_ids_are_not_invoked_twice():
    config = ModelConfig(small_model="gemini-shared", large_model="gemini-shared")
    pipeline, client, _ = setup_pipeline(
        model_config=config, replies=[sdk_response("I don't know.")],
    )
    result = pipeline.generate(QUERY)
    assert result.final_evaluation.decision == "fail"
    assert not result.escalated and "same model ID" in result.escalation_reason
    client.models.generate_content.assert_called_once_with(model="gemini-shared", contents=QUERY)


def test_complete_aggregates_include_both_attempts_even_when_first_is_selected():
    pipeline, _, records = setup_pipeline(replies=[sdk_response("First"), sdk_response("Second")])
    with (
        patch("escalation.evaluate_answer", side_effect=[quality(0.7, "fail"), quality(0.2, "fail")]),
        patch("instrumentation.time.perf_counter", side_effect=[10, 10.25, 20, 20.5]),
    ):
        result = pipeline.generate(QUERY)
    assert result.final_attempt_index == 0
    assert result.totals.input_tokens == 200
    assert result.totals.output_tokens == 40
    assert result.totals.thinking_tokens == 10
    assert result.totals.billed_output_tokens == 50
    assert result.totals.latency_ms == 750
    assert Decimal(result.totals.estimated_cost_usd) == Decimal("0.000125")
    assert result.totals.cost_status == "estimated"
    assert result.totals.attempt_cost_statuses == ("estimated", "estimated")
    for attempt, record in zip(result.attempts, records):
        assert attempt.model == record["model"]
        for field in ("input_tokens", "output_tokens", "thinking_tokens",
                      "billed_output_tokens", "estimated_cost_usd", "cost_status",
                      "timestamp_utc", "latency_ms"):
            assert getattr(attempt.response, field) == record[field]


@pytest.mark.parametrize("missing_index", [0, 1])
def test_missing_usage_prevents_partial_totals(missing_index):
    replies = [sdk_response("I don't know."), sdk_response()]
    replies[missing_index].usage_metadata = None
    pipeline, _, _ = setup_pipeline(replies=replies)
    result = pipeline.generate(QUERY)
    assert result.totals.input_tokens is result.totals.output_tokens is None
    assert result.totals.thinking_tokens is result.totals.billed_output_tokens is None
    assert result.totals.estimated_cost_usd is None and result.totals.cost_status == "incomplete"
    assert "usage_unavailable" in result.totals.attempt_cost_statuses
    assert result.totals.latency_ms is not None
    assert result.attempts[1 - missing_index].response.input_tokens == 100


@pytest.mark.parametrize("missing_model", [CONFIG.small_model, CONFIG.large_model])
def test_unpriced_attempt_prevents_partial_cost(missing_model):
    pricing = {model: RATES for model in (CONFIG.small_model, CONFIG.large_model) if model != missing_model}
    pipeline, _, _ = setup_pipeline(
        replies=[sdk_response("I don't know."), sdk_response()], pricing=pricing,
    )
    result = pipeline.generate(QUERY)
    assert result.totals.input_tokens == 200
    assert result.totals.estimated_cost_usd is None
    assert result.totals.cost_status == "pricing_unavailable"
    assert "pricing_unavailable" in result.totals.attempt_cost_statuses


def test_zero_usage_latency_and_cost_are_preserved():
    reply = sdk_response()
    reply.usage_metadata = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=0, candidates_token_count=0,
        thoughts_token_count=0, total_token_count=0,
    )
    pipeline, _, _ = setup_pipeline(replies=[reply])
    with patch("instrumentation.time.perf_counter", return_value=1):
        result = pipeline.generate(QUERY)
    assert result.totals.input_tokens == result.totals.output_tokens == result.totals.latency_ms == 0
    assert Decimal(result.totals.estimated_cost_usd) == 0
    assert result.totals.cost_status == "estimated"


def test_partial_token_fields_are_aggregated_independently():
    reply = sdk_response()
    reply.usage_metadata.prompt_token_count = None
    pipeline, _, _ = setup_pipeline(replies=[reply])
    result = pipeline.generate(QUERY)
    assert result.totals.input_tokens is None
    assert result.totals.output_tokens == 20 and result.totals.thinking_tokens == 5
    assert result.totals.estimated_cost_usd is None


def test_metric_extraction_failure_keeps_answer_and_totals_unknown():
    pipeline, _, _ = setup_pipeline()
    with patch("instrumentation.usage_metrics", side_effect=ValueError("Synthetic failure")):
        result = pipeline.generate(QUERY)
    assert result.final_response.text == "Paris."
    assert result.totals.input_tokens is result.totals.latency_ms is None
    assert result.totals.estimated_cost_usd is None
    assert result.totals.attempt_cost_statuses == ("metrics_unavailable",)


@pytest.mark.parametrize("failure_index", [0, 1])
@pytest.mark.parametrize("failure", ["api", "empty", "missing"])
def test_generation_errors_stop_and_retain_partial_results(failure_index, failure, capsys):
    bad = {"api": RuntimeError(SECRET), "empty": sdk_response(""), "missing": None}[failure]
    replies = [bad] if failure_index == 0 else [sdk_response("I don't know."), bad]
    pipeline, client, records = setup_pipeline(replies=replies)
    with pytest.raises(EscalationError) as caught:
        pipeline.generate(QUERY)
    error = caught.value
    assert error.stage == ("initial_generation" if failure_index == 0 else "escalation_generation")
    assert client.models.generate_content.call_count == error.total_attempts == failure_index + 1
    assert error.attempts[-1].response is None
    assert error.routing_decision.selected_role == "small"
    assert error.totals.input_tokens is error.totals.latency_ms is error.totals.estimated_cost_usd is None
    assert error.totals.cost_status == "incomplete"
    assert len(records) == failure_index + 1
    assert records[-1]["latency_ms"] is not None
    if failure_index:
        assert error.attempts[0].response.input_tokens == 100
        assert error.attempts[0].evaluation.decision == "fail"
        assert error.initial_routing.response is error.attempts[0].response
    else:
        assert error.initial_routing is None
    assert SECRET not in str(error)
    assert SECRET not in "".join(traceback.format_exception(error))
    assert SECRET not in repr(capsys.readouterr())


@pytest.mark.parametrize("failure_index", [0, 1])
def test_evaluation_errors_preserve_generated_metrics_and_do_not_continue(failure_index):
    pipeline, client, _ = setup_pipeline(
        replies=[sdk_response("I don't know."), sdk_response()],
    )
    effects = [RuntimeError(SECRET)] if failure_index == 0 else [quality(0.4, "fail"), RuntimeError(SECRET)]
    with patch("escalation.evaluate_answer", side_effect=effects), pytest.raises(EscalationError) as caught:
        pipeline.generate(QUERY)
    error = caught.value
    assert error.stage == ("initial_evaluation" if failure_index == 0 else "escalation_evaluation")
    assert error.total_attempts == client.models.generate_content.call_count == failure_index + 1
    assert error.attempts[-1].response is not None and error.attempts[-1].evaluation is None
    assert error.totals.input_tokens == 100 * (failure_index + 1)
    assert error.totals.estimated_cost_usd is not None
    assert error.totals.cost_status == "estimated"
    assert SECRET not in "".join(traceback.format_exception(error))


@pytest.mark.parametrize("bad", [
    None, {}, quality(float("nan"), "pass"), quality(float("inf"), "fail"),
    quality(-0.1, "fail"), quality(1.1, "pass"), quality(True, "pass"),
    quality(0.5, "unexpected"),
])
def test_malformed_evaluation_results_fail_safely(bad):
    pipeline, client, _ = setup_pipeline()
    with patch("escalation.evaluate_answer", return_value=bad), pytest.raises(EscalationError) as caught:
        pipeline.generate(QUERY)
    assert caught.value.stage == "initial_evaluation"
    client.models.generate_content.assert_called_once()


@pytest.mark.parametrize("query", [None, "", " ", 42, [], "...?!"])
def test_invalid_queries_fail_before_generation(query):
    pipeline, client, _ = setup_pipeline()
    with pytest.raises(EscalationError) as caught:
        pipeline.generate(query)
    assert caught.value.stage == "classification"
    assert caught.value.total_attempts == 0
    assert caught.value.totals.cost_status == "not_requested"
    assert caught.value.totals.estimated_cost_usd is None
    client.models.generate_content.assert_not_called()


def test_classifier_errors_are_sanitized_before_any_call():
    pipeline, client, _ = setup_pipeline()
    with patch("model_router.classify_query", side_effect=RuntimeError(SECRET)), pytest.raises(EscalationError) as caught:
        pipeline.generate(QUERY)
    assert SECRET not in "".join(traceback.format_exception(caught.value))
    assert caught.value.total_attempts == 0
    client.models.generate_content.assert_not_called()


@pytest.mark.parametrize("threshold,attempt_count", [(0.85, 1), (0.86, 2)])
def test_evaluation_threshold_is_propagated_to_both_attempts(threshold, attempt_count):
    config = EvaluationConfig(acceptance_threshold=threshold)
    pipeline, client, _ = setup_pipeline(
        replies=[sdk_response(), sdk_response("Paris is the capital of France.")],
        evaluation_config=config,
    )
    result = pipeline.generate(QUERY)
    assert result.total_attempts == client.models.generate_content.call_count == attempt_count
    assert all(attempt.evaluation.config is config for attempt in result.attempts)


def test_invalid_pipeline_configuration():
    with pytest.raises(ValueError, match="ModelRouter"):
        EscalationPipeline(None)
    pipeline, client, _ = setup_pipeline()
    with pytest.raises(ValueError, match="EvaluationConfig"):
        EscalationPipeline(pipeline.router, evaluation_config={})
    client.models.generate_content.assert_not_called()


def test_results_are_immutable_and_selection_is_deterministic():
    results = []
    for _ in range(2):
        pipeline, _, _ = setup_pipeline(replies=[sdk_response("I don't know."), sdk_response()])
        with patch("instrumentation.time.perf_counter", side_effect=[1, 1.25, 2, 2.5]):
            results.append(pipeline.generate(QUERY))
    first, second = results
    assert first.final_attempt_index == second.final_attempt_index
    assert first.initial_routing.decision == second.initial_routing.decision
    assert first.escalation_reason == second.escalation_reason
    assert first.final_evaluation == second.final_evaluation
    assert first.totals == second.totals
    with pytest.raises(FrozenInstanceError):
        first.escalated = False
