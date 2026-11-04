"""Synthetic offline quality checks; no generation or factual verification."""

from dataclasses import FrozenInstanceError, replace
import socket

import pytest

from answer_evaluator import evaluate_answer
from evaluation_config import EvaluationConfig, DEFAULT_WEIGHTS


QUERY = "What is the capital of France?"


def criterion(result, name):
    return next(item for item in result.criteria if item.name == name)


@pytest.mark.parametrize("answer", ["", "  ", "\n", "...", "???", "TODO", "N/A", "[answer]", "Okay."])
def test_empty_or_placeholder_always_fails_even_at_zero_threshold(answer):
    result = evaluate_answer(QUERY, answer, config=EvaluationConfig(acceptance_threshold=0))
    assert result.decision == "fail" and result.score == 0
    assert result.blockers == ("missing_answer",)


@pytest.mark.parametrize("query,answer", [
    (QUERY, "Paris."),
    ("Calculate 2 + 2.", "4"),
    ("Is a Python tuple immutable?", "Yes."),
    ("Define recursion.", "Recursion is a process that calls itself."),
    ("What is a tuple?", "A tuple is an immutable sequence."),
])
def test_short_complete_answers_can_pass(query, answer):
    result = evaluate_answer(query, answer)
    assert result.decision == "pass"
    assert not result.factual_correctness_verified


def test_short_answer_relevance_is_explicitly_uncertain():
    result = evaluate_answer(QUERY, "Paris.")
    assert criterion(result, "relevance").score == 0.5
    assert "cannot be established" in criterion(result, "relevance").reason
    assert result.score == 0.85


def test_short_wrong_answer_is_not_mistaken_for_verified_truth():
    # This intentional weakness demonstrates that acceptance is not fact checking.
    result = evaluate_answer("Calculate 2 + 2.", "9")
    assert result.decision == "pass" and not result.factual_correctness_verified


@pytest.mark.parametrize("answer", [
    "Grass grows in fields and needs water.",
    "Ocean waves carry energy across water. " * 100,
])
def test_irrelevant_answers_fail_regardless_of_length(answer):
    result = evaluate_answer("Explain Python tuples.", answer)
    assert criterion(result, "relevance").score == 0
    assert result.decision == "fail"


def test_repeating_relevant_content_does_not_increase_quality():
    short = evaluate_answer("Define recursion.", "Recursion calls itself.")
    repeated = evaluate_answer("Define recursion.", "Recursion calls itself. " * 20)
    assert repeated.score < short.score
    assert criterion(repeated, "structure").score == 0


def test_structured_list_passes_and_missing_items_fail():
    query = "List three benefits of tuples."
    complete = "- Tuples are immutable.\n- Tuples support unpacking.\n- Tuples can hold mixed types."
    incomplete = "- Tuples are immutable.\n- Tuples support unpacking."
    good, bad = evaluate_answer(query, complete), evaluate_answer(query, incomplete)
    assert good.decision == "pass"
    assert bad.decision == "fail" and "incomplete_answer" in bad.blockers
    assert criterion(bad, "completeness").score == pytest.approx(2 / 3)


def test_repeated_list_items_do_not_count_as_multiple_items():
    result = evaluate_answer("Give three benefits of tuples.", "- Tuples are immutable.\n" * 3)
    assert result.decision == "fail"
    assert criterion(result, "completeness").score == pytest.approx(1 / 3)


@pytest.mark.parametrize("ending", ["...", ":", "…", " TODO"])
def test_truncated_or_unfinished_answers_fail(ending):
    result = evaluate_answer("Explain recursion.", "Recursion works as follows" + ending)
    assert result.decision == "fail"
    assert "incomplete_answer" in result.blockers


def test_code_requirement_is_checked_without_executing_code():
    query = "Write a Python function to add two numbers."
    assert evaluate_answer(query, "A Python function can add numbers.").decision == "fail"
    # A prose promise does not satisfy the recognized code-output requirement.
    result = evaluate_answer(query, "I can do that.")
    assert result.decision == "fail"
    assert "incomplete_answer" in result.blockers
    code = "```python\ndef add(a, b):\n    return a + b\n```"
    assert criterion(evaluate_answer(query, code), "completeness").score == 1


@pytest.mark.parametrize("answer,expected", [
    ('{"color": "blue"}', "pass"), ('```json\n{"color": "blue"}\n```', "pass"),
    ('{"color": blue}', "fail"),
])
def test_requested_json_syntax(answer, expected):
    result = evaluate_answer("Return JSON with a color field.", answer)
    assert result.decision == expected


def test_unclosed_code_fence_and_paragraph_readability():
    result = evaluate_answer("Explain Python.", "```python\nPython code")
    assert criterion(result, "structure").score == 0
    result = evaluate_answer("Explain Python.", "Python is readable and useful.",
                             config=EvaluationConfig(max_paragraph_chars=10))
    assert criterion(result, "structure").score == 0


@pytest.mark.parametrize("answer", [
    "I don't know the capital of France.",
    "I am not sure which city is the capital of France.",
    "Perhaps Paris is the capital of France.",
    "I cannot answer the question about France.",
])
def test_uncertainty_is_explicit_and_blocks_by_default(answer):
    result = evaluate_answer(QUERY, answer)
    assert result.decision == "fail" and "explicit_uncertainty" in result.blockers
    assert criterion(result, "certainty").score == 0


@pytest.mark.parametrize("answer", [
    "Error: Gemini generation failed.", "Internal server error",
    "Service unavailable", "Generation failed", "Request failed",
    "Rate limit exceeded", "Quota exceeded", '{"error": "synthetic failure"}',
])
def test_failure_messages_block_even_at_zero_threshold(answer):
    result = evaluate_answer(QUERY, answer, config=EvaluationConfig(acceptance_threshold=0))
    assert result.decision == "fail" and "generation_error" in result.blockers


def test_explaining_errors_is_not_a_leading_generation_failure():
    result = evaluate_answer("Explain a Python error.",
                             "A Python error indicates that an operation could not complete.")
    assert criterion(result, "error_free").score == 1
    assert result.decision == "pass"


@pytest.mark.parametrize("threshold,decision", [(0.849999, "pass"), (0.85, "pass"), (0.850001, "fail")])
def test_threshold_boundary_uses_reported_score(threshold, decision):
    result = evaluate_answer(QUERY, "Paris.", config=EvaluationConfig(acceptance_threshold=threshold))
    assert result.score == 0.85 and result.decision == decision
    assert result.acceptance_threshold == threshold


def test_strict_and_permissive_thresholds():
    assert evaluate_answer(QUERY, "Paris.", config=EvaluationConfig(acceptance_threshold=1)).decision == "fail"
    assert evaluate_answer(QUERY, "France has Paris as its capital.",
                           config=EvaluationConfig(acceptance_threshold=1)).decision == "pass"


def test_blocking_settings_are_configurable():
    config = EvaluationConfig(acceptance_threshold=0, fail_on_error=False,
                              fail_on_uncertainty=False, fail_on_incomplete=False)
    result = evaluate_answer(QUERY, "Error: I don't know France...", config=config)
    assert result.decision == "pass" and not result.blockers
    assert result.score < 1


@pytest.mark.parametrize("query,answer", [
    (None, "answer"), ("", "answer"), ("?!", "answer"), (2, "answer"),
    (QUERY, None), (QUERY, 3), (QUERY, []), (QUERY, b"answer"),
])
def test_invalid_inputs(query, answer):
    with pytest.raises(ValueError):
        evaluate_answer(query, answer)


@pytest.mark.parametrize("field,value", [
    ("acceptance_threshold", -0.1), ("acceptance_threshold", 1.1),
    ("acceptance_threshold", True), ("acceptance_threshold", "0.8"),
    ("acceptance_threshold", float("nan")), ("acceptance_threshold", float("inf")),
    ("short_answer_max_words", 0), ("short_answer_max_words", True),
    ("max_paragraph_chars", -1), ("max_paragraph_chars", 1.5),
    ("fail_on_error", 1), ("fail_on_uncertainty", None), ("fail_on_incomplete", "yes"),
    ("weights", []), ("weights", (("unknown", 1),)), ("weights", (([], 1),)),
    ("weights", DEFAULT_WEIGHTS + (("presence", 1),)),
])
def test_invalid_configuration(field, value):
    with pytest.raises(ValueError):
        EvaluationConfig(**{field: value})


@pytest.mark.parametrize("weight", [-1, True, 0.5, "1"])
def test_invalid_weight_values(weight):
    weights = tuple((name, weight if name == "presence" else value) for name, value in DEFAULT_WEIGHTS)
    with pytest.raises(ValueError):
        EvaluationConfig(weights=weights)


def test_zero_weights_and_wrong_config_rejected():
    with pytest.raises(ValueError):
        EvaluationConfig(weights=tuple((name, 0) for name, _ in DEFAULT_WEIGHTS))
    with pytest.raises(ValueError):
        evaluate_answer(QUERY, "Paris.", config={})


def test_results_are_deterministic_immutable_and_do_not_retain_text():
    config = EvaluationConfig()
    expected = evaluate_answer(QUERY, "Paris.", config=config)
    for _ in range(5):
        assert evaluate_answer(QUERY, "Paris.", config=config) == expected
    assert evaluate_answer(QUERY, "Paris.", config=replace(config, weights=tuple(reversed(config.weights)))).score == expected.score
    with pytest.raises(FrozenInstanceError):
        expected.decision = "fail"
    assert "Paris" not in repr(expected) and QUERY not in repr(expected)


def test_offline_with_network_disabled(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Network is forbidden")
    monkeypatch.setattr(socket, "create_connection", fail)
    monkeypatch.setattr(socket.socket, "connect", fail)
    assert evaluate_answer(QUERY, "Paris.").decision == "pass"
