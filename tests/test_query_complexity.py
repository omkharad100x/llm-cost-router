"""Offline behavior, validation, and boundary tests for query complexity."""

from dataclasses import FrozenInstanceError, replace
import socket

import pytest

from complexity_config import ComplexityConfig, DEFAULT_WEIGHTS
from query_complexity import classify_query


@pytest.mark.parametrize("query", [
    "What is the capital of France?",
    "Who wrote Hamlet?",
    "Define recursion.",
    "What is Python code?",
    "What does the word debug mean?",
    "What is an integral?",
    "Calculate 2 + 2.",
    "Translate hello into Spanish.",
    "List apples, oranges, and bananas.",
    "Could you please tell me the capital of France?",
])
def test_simple_questions(query):
    result = classify_query(query)
    assert result.label == "simple"
    assert result.score < 3 and result.reasons


def test_long_straightforward_request_is_not_complex_from_length_alone():
    query = ("Please use the familiar English name and keep the answer brief. " * 20
             + "What is the capital of France?")
    result = classify_query(query)
    assert result.word_count >= 120
    assert result.label == "simple"
    assert [signal.name for signal in result.signals] == ["length"]


@pytest.mark.parametrize("query,signal", [
    ("First calculate the total, then explain the result.", "multi_step"),
    ("Explain the solution step-by-step.", "multi_step"),
    ("Prove that there are infinitely many prime numbers.", "reasoning"),
    ("Derive a recurrence relation for this sequence.", "reasoning"),
    ("Justify the conclusion using a counterexample.", "reasoning"),
    ("Write a Python function to merge two sorted lists.", "coding"),
    ("Generate SQL to join the customer and order tables.", "coding"),
    ("Debug this function that raises an IndexError.", "coding"),
    ("Can you fix this Python code error?", "coding"),
    ("Please refactor this function to reduce duplication.", "coding"),
    ("Solve the quadratic equation x^2 - 5*x + 6 = 0.", "advanced_math"),
    ("Calculate the probability of exactly two heads in five tosses.", "advanced_math"),
    ("Differentiate x^3 with respect to x.", "advanced_math"),
    ("Analyze the dataset to identify trends.", "analysis"),
    ("Compare the performance trade-offs of these algorithms.", "analysis"),
    ("Evaluate the risks of these designs.", "analysis"),
])
def test_complex_work_has_explicit_evidence(query, signal):
    result = classify_query(query)
    assert result.label == "complex"
    assert any(item.name == signal for item in result.signals)


@pytest.mark.parametrize("query", [
    "Summarize the article and translate the summary into French.",
    "Define a tuple; list three examples.",
    "What is a tuple? How is a list different?",
    "1. Describe a tuple.\n2. Compare lists and tuples.",
    "- Summarize the passage\n- Translate the summary",
])
def test_multiple_tasks(query):
    result = classify_query(query)
    assert result.label == "complex"
    assert result.instruction_count >= 2
    assert "multiple_tasks" in [signal.name for signal in result.signals]


def test_context_and_explanation_are_separate_weak_signals():
    result = classify_query("Using the following passage, explain in detail its main idea.")
    assert result.label == "simple" and result.score == 2
    assert {signal.name for signal in result.signals} == {"context", "explanation"}
    config = ComplexityConfig(long_query_words=5)
    assert classify_query(
        "Using the following passage, explain in detail its main idea.", config=config,
    ).label == "complex"


def test_repetition_is_not_counted_as_distinct_tasks():
    result = classify_query("Define a tuple. Define a tuple.")
    assert result.instruction_count == 1 and result.label == "simple"


@pytest.mark.parametrize("query", ["", "  \n", None, 123, True, [], {}, b"query", "...?!"])
def test_invalid_queries(query):
    with pytest.raises(ValueError, match="Query"):
        classify_query(query)


@pytest.mark.parametrize("field,value", [
    ("complex_threshold", 0), ("complex_threshold", -1),
    ("complex_threshold", True), ("complex_threshold", 2.5),
    ("complex_threshold", float("nan")), ("complex_threshold", float("inf")),
    ("long_query_words", 0), ("long_query_words", "120"),
    ("multiple_tasks_min", 1), ("multiple_tasks_min", False),
    ("weights", list(DEFAULT_WEIGHTS)),
    ("weights", (("unknown", 1),)),
    ("weights", DEFAULT_WEIGHTS + (("length", 1),)),
    ("weights", (([], 1),)),
    ("weights", (("length",),)),
])
def test_invalid_configuration(field, value):
    with pytest.raises(ValueError):
        ComplexityConfig(**{field: value})


@pytest.mark.parametrize("weight", [-1, True, 1.5, float("inf"), "3"])
def test_invalid_signal_weights(weight):
    weights = tuple((name, weight if name == "coding" else score)
                    for name, score in DEFAULT_WEIGHTS)
    with pytest.raises(ValueError):
        ComplexityConfig(weights=weights)


@pytest.mark.parametrize("config", [None, {}, 1])
def test_invalid_configuration_object(config):
    with pytest.raises(ValueError, match="ComplexityConfig"):
        classify_query("What is a tuple?", config=config)


@pytest.mark.parametrize("threshold,expected", [(2, "complex"), (3, "complex"), (4, "simple")])
def test_score_threshold_is_inclusive(threshold, expected):
    config = ComplexityConfig(complex_threshold=threshold)
    result = classify_query("Prove this statement.", config=config)
    assert result.score == 3 and result.label == expected


@pytest.mark.parametrize("words,expected", [(2, 0), (3, 1), (4, 1)])
def test_word_count_boundary(words, expected):
    result = classify_query(" ".join(["hello"] * words),
                            config=ComplexityConfig(long_query_words=3))
    assert result.score == expected and result.label == "simple"


def test_instruction_count_threshold_is_configurable():
    query = "Define a tuple and list examples."
    assert classify_query(query).label == "complex"
    config = ComplexityConfig(multiple_tasks_min=3)
    assert classify_query(query, config=config).label == "simple"


def test_rule_weight_can_be_disabled_without_changing_others():
    weights = tuple((name, 0 if name == "coding" else weight)
                    for name, weight in DEFAULT_WEIGHTS)
    result = classify_query("Write a Python function.", config=ComplexityConfig(weights=weights))
    assert result.label == "simple" and result.score == 0
    assert all(signal.name != "coding" for signal in result.signals)


def test_identical_input_and_config_are_deterministic_and_immutable():
    config = ComplexityConfig()
    expected = classify_query("Write a Python function and justify its complexity.", config=config)
    for _ in range(10):
        assert classify_query("Write a Python function and justify its complexity.", config=config) == expected
    reversed_config = replace(config, weights=tuple(reversed(config.weights)))
    assert classify_query("Write a Python function and justify its complexity.",
                          config=reversed_config) == expected
    with pytest.raises(FrozenInstanceError):
        expected.label = "simple"
    with pytest.raises(FrozenInstanceError):
        config.complex_threshold = 1


def test_case_and_outer_whitespace_do_not_change_classification():
    assert classify_query("  PROVE this statement. \n") == classify_query("Prove this statement.")


def test_signals_are_counted_at_most_once():
    result = classify_query("Prove, derive, justify, and deduce the conclusion.")
    assert sum(signal.name == "reasoning" for signal in result.signals) == 1


def test_classification_works_with_network_disabled(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Network access is forbidden in offline classification")

    monkeypatch.setattr(socket, "create_connection", fail)
    monkeypatch.setattr(socket.socket, "connect", fail)
    assert classify_query("Write a Python script.").label == "complex"
