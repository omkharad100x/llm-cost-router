"""Hand-calculated offline benchmark checks; all sockets are blocked."""

from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path
import socket

import pytest

from benchmarks import (
    BenchmarkConfig, DEFAULT_FIXTURES, DEFAULT_PRICING,
    cache_metrics, format_summary, load_fixtures, main, run_benchmarks, summarize_requests,
)
from evaluate_similarity import load_dataset
from model_config import ModelConfig


@pytest.fixture(scope="module", autouse=True)
def offline():
    with pytest.MonkeyPatch.context() as monkeypatch:
        def blocked(*args, **kwargs):
            raise AssertionError("Network is forbidden.")
        monkeypatch.setattr(socket, "create_connection", blocked)
        monkeypatch.setattr(socket.socket, "connect", blocked)
        monkeypatch.setattr(socket.socket, "connect_ex", blocked)
        yield


@pytest.fixture(scope="module")
def report(offline):
    return run_benchmarks()


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def fixtures():
    return json.loads(DEFAULT_FIXTURES.read_text(encoding="utf-8"))


def record(model="gemini-small", cost="0.2", status="estimated", tokens=10, latency=2.0):
    return {"model": model, "input_tokens": tokens, "output_tokens": tokens,
            "thinking_tokens": 0, "billed_output_tokens": tokens, "latency_ms": latency,
            "estimated_cost_usd": cost, "cost_status": status}


def without_timings(value):
    if isinstance(value, dict):
        return {key: without_timings(item) for key, item in value.items() if key != "latency"}
    if isinstance(value, (list, tuple)):
        return [without_timings(item) for item in value]
    return value


def test_confusion_matrix_and_denominators():
    metrics = cache_metrics([
        {"expected": "SAFE_MATCH", "matched": True},
        {"expected": "SAFE_MATCH", "matched": False},
        {"expected": "SAFE_MATCH", "matched": False},
        {"expected": "MUST_NOT_MATCH", "matched": True},
        {"expected": "MUST_NOT_MATCH", "matched": False},
    ])
    assert (metrics["true_hits"], metrics["false_hits"], metrics["true_misses"],
            metrics["false_misses"]) == (1, 1, 1, 2)
    assert metrics["precision"] == 1 / 2
    assert metrics["recall"] == 1 / 3
    assert metrics["false_hit_rate"] == 1 / 2
    assert metrics["false_miss_rate"] == 2 / 3
    assert metrics["hit_rate"] == 2 / 5 and metrics["miss_rate"] == 3 / 5


@pytest.mark.parametrize("index,counts", [
    (0, (8, 11, 3, 0)), (1, (6, 6, 8, 2)), (2, (3, 4, 10, 5)),
])
def test_cache_threshold_comparisons_have_expected_counts(report, index, counts):
    experiment = report["cache_benchmarks"][index]
    metrics = experiment["metrics"]
    assert tuple(metrics[key] for key in ("true_hits", "false_hits", "true_misses", "false_misses")) == counts
    assert experiment["eligible_pairs"] == 21
    assert sum(counts) == experiment["pair_count"] == 22
    assert metrics["false_hit_rate"] == counts[1] / 14
    assert metrics["false_miss_rate"] == counts[3] / 8


def test_reuse_and_actual_generation_calls_are_counted_separately(report):
    experiment = report["cache_benchmarks"][1]
    assert experiment["reuse"] == {
        "lookup_count": 44, "generation_calls": 20, "generation_calls_avoided": 24,
        "safe_reuse_count": 12, "unsafe_reuse_count": 12,
        "repeat_hits_after_first_lookup": 12,
    }
    assert experiment["generation_metrics"]["requests"] == 20
    assert experiment["embedding_metrics"]["requests"] == 42
    assert experiment["setup_metrics"]["requests"] == 42  # 21 generation + 21 embedding
    assert experiment["request_metrics"]["requests"] == 62
    assert experiment["request_metrics"]["cost"]["estimated_cost_usd"] is None
    assert experiment["embedding_metrics"]["usage"]["input_tokens"] is None


def test_eligibility_denial_is_separate_from_similarity(report):
    cache = report["cache_benchmarks"][1]
    denied = next(row for row in cache["rows"] if row["id"] == "entity_capital")
    assert not denied["eligible"] and not denied["seeded"] and not denied["matched"]
    assert denied["lookup_reason"] == "ineligible"
    similarity = report["similarity_only"][1]
    assert similarity["metrics"]["false_positives"] == 7
    assert cache["metrics"]["false_hits"] == 6
    assert cache["eligible_metrics"]["false_hit_rate"] == 6 / 13


def test_routing_distribution_comes_from_real_classifier_and_router(report):
    routing = report["pipeline_benchmark"]["routing_metrics"]
    assert routing["query_count"] == 8
    assert routing["classifications"] == {"simple": 6, "complex": 2}
    assert routing["initial_selections"] == {"small": 6, "large": 2}
    assert routing["routing_distribution"] == {"small": 0.75, "large": 0.25}
    assert routing["all_attempt_roles"] == {"small": 6, "large": 6}
    assert routing["classification_fixture_agreement"] == 1
    cases = report["pipeline_benchmark"]["cases"]
    assert all(case["classification"]["reasons"] for case in cases)
    assert all(case["routing_decision"]["cost_optimized"] is False for case in cases)


def test_escalation_frequency_success_and_attempts(report):
    assert report["pipeline_benchmark"]["escalation_metrics"] == {
        "initial_answers_accepted": 3, "escalations_triggered": 4,
        "escalation_rate": 0.5, "successful_escalations": 1,
        "escalation_success_rate": 0.25, "both_answers_fail": 3,
        "total_model_attempts": 12, "average_attempts_per_query": 1.5,
    }
    cases = {case["id"]: case for case in report["pipeline_benchmark"]["cases"]}
    assert cases["both_fail_tie"]["final_attempt_index"] == 0
    assert cases["both_fail_second_better"]["final_attempt_index"] == 1
    assert cases["both_fail_initial_better"]["final_attempt_index"] == 0
    assert len(cases["complex_fail"]["attempts"]) == 1


def test_quality_metrics_are_heuristic_fixture_agreement(report):
    quality = report["pipeline_benchmark"]["quality_metrics"]
    assert quality["attempt_passes"] == quality["final_passes"] == 4
    assert quality["attempt_failures"] == 8
    assert quality["fixture_label_agreement"] == 1
    assert quality["factual_correctness_verified"] is False


def test_known_cost_aggregation_by_model_and_denominator():
    result = summarize_requests([record(), record(cost="0.1"), record("gemini-large", cost="0.9")],
                                query_count=2)
    assert Decimal(result["cost"]["estimated_cost_usd"]) == Decimal("1.2")
    assert Decimal(result["cost"]["average_cost_per_query_usd"]) == Decimal("0.6")
    assert result["cost"]["complete"]
    assert Decimal(result["by_model"]["gemini-small"]["cost"]["estimated_cost_usd"]) == Decimal("0.3")
    assert result["usage"]["input_tokens"] == result["usage"]["output_tokens"] == 30


@pytest.mark.parametrize("status", ["pricing_unavailable", "usage_unavailable", "metrics_unavailable"])
def test_unknown_cost_is_not_zero_or_partial_total(status):
    result = summarize_requests([record(), record(cost=None, status=status)], query_count=2)
    assert result["cost"]["estimated_cost_usd"] is None
    assert result["cost"]["average_cost_per_query_usd"] is None
    assert result["cost"]["known_cost_subtotal_usd"] == "0.2"
    assert result["cost"]["unknown_cost_requests"] == 1 and not result["cost"]["complete"]
    assert result["cost"]["status"] == ("pricing_unavailable" if status == "pricing_unavailable" else "incomplete")


def test_default_report_cost_is_incomplete_and_does_not_guess_large_rates(report):
    result = report["pipeline_benchmark"]["request_metrics"]
    assert result["cost"]["status"] == "pricing_unavailable"
    assert result["cost"]["estimated_cost_usd"] is None
    assert result["cost"]["unknown_cost_requests"] == 7
    assert Decimal(result["cost"]["known_cost_subtotal_usd"]) == Decimal("0.0003125")
    assert result["usage"]["input_tokens"] is result["usage"]["output_tokens"] is None
    assert report["configuration"]["models"]["large_model"] not in report["configuration"]["pricing"]


def test_missing_usage_and_latency_are_aggregated_independently():
    partial = record(tokens=None, latency=None)
    partial["output_tokens"] = 3
    result = summarize_requests([record(), partial], query_count=2)
    assert result["usage"]["input_tokens"] is None
    assert result["usage"]["output_tokens"] == 13
    assert result["latency"]["total_ms"] is result["latency"]["average_per_request_ms"] is None


def test_latency_aggregation_and_zero_values():
    result = summarize_requests([record(latency=2.5), record(latency=7.5)], query_count=4)
    assert result["latency"]["total_ms"] == 10
    assert result["latency"]["average_per_request_ms"] == 5
    assert result["latency"]["average_per_query_ms"] == 2.5
    zero = summarize_requests([record(cost="0", tokens=0, latency=0)], query_count=1)
    assert zero["cost"]["estimated_cost_usd"] == "0"
    assert zero["usage"]["input_tokens"] == zero["latency"]["total_ms"] == 0


def test_measured_lookup_latency_is_separate_and_finite(report):
    for experiment in report["cache_benchmarks"]:
        count = experiment["reuse"]["lookup_count"]
        assert experiment["latency"]["total_lookup_ms"] >= 0
        assert experiment["latency"]["average_lookup_ms"] == experiment["latency"]["total_lookup_ms"] / count
        assert "generation excluded" in experiment["latency"]["source"]
        assert "not provider" in experiment["generation_metrics"]["latency"]["source"]


def test_json_report_validity_and_configuration_snapshot(report):
    parsed = json.loads(json.dumps(report, allow_nan=False))
    assert parsed["report_schema_version"] == 1
    assert parsed["configuration"]["thresholds"] == [0.8, 0.9, 0.95]
    assert len(parsed["datasets"]["sha256"]) == 4
    assert all(len(value) == 64 for value in parsed["datasets"]["sha256"].values())
    assert "not Gemini" in parsed["datasets"]["embedding_source"]
    assert "not production" in format_summary(report)
    assert len(parsed["measurement_notes"]) >= 10


def test_repeated_runs_have_identical_non_timing_metrics(report):
    assert without_timings(report) == without_timings(run_benchmarks())


def test_complete_synthetic_fixture_costs_can_be_calculated_without_default_changes(tmp_path):
    data = fixtures()
    for case in data["routing_cases"]:
        for reply in case["responses"].values():
            if reply["usage"] is None:
                reply["usage"] = data["cache_generation_usage"]
    data["embedding_token_count"] = 10  # Explicit mock statistics for this test only.
    fixture_path = write_json(tmp_path / "fixtures.json", data)
    rates = json.loads(DEFAULT_PRICING.read_text(encoding="utf-8"))
    rates[ModelConfig().large_model] = {
        "input_usd_per_million": "1", "output_usd_per_million": "2",
        "pricing_version": "synthetic-test-rates-not-production",
    }
    rate_path = write_json(tmp_path / "rates.json", rates)
    result = run_benchmarks(config=BenchmarkConfig(thresholds=(0.9,), lookup_repeats=1),
                            fixtures=fixture_path, pricing=rate_path)
    metrics = result["pipeline_benchmark"]["request_metrics"]
    assert Decimal(metrics["cost"]["estimated_cost_usd"]) == Decimal("0.001275")
    assert metrics["cost"]["complete"]
    assert metrics["usage"]["input_tokens"] == 1200
    assert metrics["usage"]["output_tokens"] == 240
    assert result["cache_benchmarks"][0]["request_metrics"]["cost"]["complete"]


def test_empty_datasets_return_undefined_rates_and_zero_requests(tmp_path):
    data = fixtures()
    data["routing_cases"] = []
    data["cache_reviews"] = {}
    fixture_path = write_json(tmp_path / "fixtures.json", data)
    dataset_path = write_json(tmp_path / "pairs.json", {"schema_version": 1, "pairs": []})
    vector_path = write_json(tmp_path / "vectors.json", {"schema_version": 1, "source": "empty synthetic", "embeddings": {}})
    result = run_benchmarks(fixtures=fixture_path, dataset=dataset_path, embeddings=vector_path)
    for experiment in result["cache_benchmarks"]:
        assert experiment["metrics"]["hit_rate"] is experiment["metrics"]["false_hit_rate"] is None
        assert experiment["reuse"]["generation_calls"] == 0
        assert experiment["latency"]["average_lookup_ms"] is None
    pipeline = result["pipeline_benchmark"]
    assert pipeline["escalation_metrics"]["escalation_success_rate"] is None
    assert pipeline["escalation_metrics"]["average_attempts_per_query"] is None
    assert pipeline["request_metrics"]["cost"]["estimated_cost_usd"] == "0"
    assert pipeline["request_metrics"]["cost"]["average_cost_per_query_usd"] is None


@pytest.mark.parametrize("kwargs", [
    {"thresholds": ()}, {"thresholds": [0.9]}, {"thresholds": (0.9, 0.9)},
    {"thresholds": (True,)}, {"thresholds": (float("nan"),)},
    {"thresholds": (float("inf"),)}, {"thresholds": (1.1,)}, {"thresholds": (-1.1,)},
    {"thresholds": ("0.9",)}, {"lookup_repeats": 0}, {"lookup_repeats": -1},
    {"lookup_repeats": True}, {"lookup_repeats": 1.5}, {"models": {}},
    {"routing": {}}, {"evaluation": {}},
    {"models": ModelConfig(small_model="gemini-shared", large_model="gemini-shared")},
])
def test_invalid_benchmark_config(kwargs):
    with pytest.raises(ValueError):
        BenchmarkConfig(**kwargs)


def test_threshold_endpoints_and_non_default_repeats():
    result = run_benchmarks(config=BenchmarkConfig(thresholds=(-1, 1), lookup_repeats=1))
    assert result["cache_benchmarks"][0]["reuse"]["generation_calls_avoided"] == 21
    assert result["cache_benchmarks"][1]["metrics"]["false_hits"] == 0
    assert result["cache_benchmarks"][0]["reuse"]["lookup_count"] == 22


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(schema_version=True),
    lambda d: d["cache_reviews"].pop("safe_tuple"),
    lambda d: d["cache_reviews"]["safe_tuple"].update(query_a="Changed query"),
    lambda d: d["cache_reviews"]["safe_tuple"]["query_flags"].update(non_sensitive=1),
    lambda d: d["cache_reviews"]["safe_tuple"].update(response_flags={}),
    lambda d: d["routing_cases"].append(deepcopy(d["routing_cases"][0])),
    lambda d: d["routing_cases"][0].update(query=""),
    lambda d: d["routing_cases"][0].update(query="...?!"),
    lambda d: d["routing_cases"][0]["responses"]["small"].update(usage={"prompt_token_count": -1}),
    lambda d: d["routing_cases"][0]["responses"]["small"].update(expected_quality="maybe"),
    lambda d: d.update(embedding_token_count=-1),
    lambda d: d.update(cache_generation_usage={}),
])
def test_invalid_fixtures_are_rejected_before_workload(tmp_path, mutation):
    data = fixtures()
    mutation(data)
    path = write_json(tmp_path / "invalid.json", data)
    with pytest.raises(ValueError):
        load_fixtures(path, load_dataset())


def test_missing_vectors_and_invalid_normalization_fail_before_benchmark(tmp_path):
    from evaluate_similarity import DEFAULT_EMBEDDINGS
    data = json.loads(DEFAULT_EMBEDDINGS.read_text(encoding="utf-8"))
    data["embeddings"].pop("What is a Python tuple?")
    path = write_json(tmp_path / "missing.json", data)
    with pytest.raises(ValueError, match="Missing embedding"):
        run_benchmarks(embeddings=path)
    data["embeddings"]["What is a Python tuple?"] = [100, 0]
    write_json(path, data)
    with pytest.raises(ValueError, match="normalized"):
        run_benchmarks(embeddings=path)


@pytest.mark.parametrize("field,value", [
    ("input_tokens", -1), ("output_tokens", True),
    ("latency_ms", float("nan")), ("latency_ms", -1),
    ("estimated_cost_usd", "NaN"), ("estimated_cost_usd", "-0.1"),
])
def test_invalid_metric_records(field, value):
    row = record()
    row[field] = value
    with pytest.raises(ValueError):
        summarize_requests([row], query_count=1)


def test_invalid_query_count_and_config():
    with pytest.raises(ValueError):
        summarize_requests([], query_count=True)
    with pytest.raises(ValueError):
        run_benchmarks(config={})


def test_cli_writes_json_and_summary_and_preserves_existing_files(tmp_path, capsys):
    path = tmp_path / "report.json"
    assert main(["--thresholds", "0.9", "--lookup-repeats", "1", "--output", str(path)]) == 0
    original = path.read_bytes()
    assert json.loads(original)["configuration"]["lookup_repeats"] == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "OFFLINE SYNTHETIC" in captured.err
    with pytest.raises(SystemExit) as caught:
        main(["--output", str(path)])
    assert caught.value.code == 2 and path.read_bytes() == original


def test_cli_stdout_is_machine_readable_json(capsys):
    assert main(["--thresholds", "0.9"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["datasets"]["pair_count"] == 22
    assert "OFFLINE SYNTHETIC" in captured.err


@pytest.mark.parametrize("arguments", [["--thresholds", "nan"], ["--lookup-repeats", "0"]])
def test_invalid_cli_inputs(arguments):
    with pytest.raises(SystemExit) as caught:
        main(arguments)
    assert caught.value.code == 2


def test_pair_namespaces_and_mock_responses_are_isolated_for_shared_text(tmp_path):
    from evaluate_similarity import load_embeddings
    pairs = load_dataset()[:2]
    pairs[1]["query_a"] = pairs[0]["query_a"]
    pairs[1]["query_b"] = "Define the Python tuple data type."
    data = fixtures()
    data["routing_cases"] = []
    data["cache_reviews"] = {pair["id"]: data["cache_reviews"][pair["id"]] for pair in pairs}
    second_review = data["cache_reviews"][pairs[1]["id"]]
    second_review.update(query_a=pairs[1]["query_a"], query_b=pairs[1]["query_b"],
                         response="A tuple in Python is an immutable sequence.")
    _, vectors = load_embeddings()
    vectors[pairs[1]["query_b"]] = [1.0, 0.0]
    result = run_benchmarks(
        config=BenchmarkConfig(thresholds=(0.9,), lookup_repeats=1),
        dataset=write_json(tmp_path / "pairs.json", {"schema_version": 1, "pairs": pairs}),
        embeddings=write_json(tmp_path / "vectors.json",
                              {"schema_version": 1, "source": "synthetic", "embeddings": vectors}),
        fixtures=write_json(tmp_path / "fixtures.json", data),
    )
    assert all(row["seeded"] and row["matched"] for row in result["cache_benchmarks"][0]["rows"])
    assert result["cache_benchmarks"][0]["reuse"]["generation_calls"] == 0
