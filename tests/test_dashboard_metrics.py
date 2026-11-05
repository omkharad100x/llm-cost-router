"""Offline Task 2 projections using only temporary reports and request logs."""

import json

import pytest

from dashboard_data import load_benchmark, load_runtime


@pytest.fixture
def report():
    return {
        "report_schema_version": 1,
        "configuration": {"thresholds": [0.9]},
        "datasets": {"pair_count": 4, "routing_case_count": 2},
        "cache_benchmarks": [{
            "threshold": 0.9, "pair_count": 4,
            "metrics": {"true_hits": 1, "false_hits": 1, "true_misses": 2, "false_misses": 0},
            "reuse": {"lookup_count": 12, "generation_calls": 6, "generation_calls_avoided": 6},
            "latency": {"average_lookup_ms": 0},
            "rows": [{"id": str(i), "seeded": i != 0} for i in range(4)],
        }],
        "pipeline_benchmark": {
            "routing_metrics": {"query_count": 2, "classifications": {"simple": 2, "complex": 0},
                                "initial_selections": {"small": 2, "large": 0},
                                "all_attempt_roles": {"small": 2, "large": 1}},
            "cases": [{"id": "a", "escalated": True, "final_attempt_index": 0,
                       "attempts": [{"role": "small"}, {"role": "large"}]},
                      {"id": "b", "final_attempt_index": 0, "attempts": [{"role": "small"}]}],
            "escalation_metrics": {"total_model_attempts": 3, "escalations_triggered": 1,
                                   "initial_answers_accepted": 1, "successful_escalations": 0},
        },
    }


def load(directory, payload):
    (directory / "benchmark_report.json").write_text(json.dumps(payload), encoding="utf-8")
    return load_benchmark(directory)


def test_pair_workload_and_storage_have_different_units(tmp_path, report):
    cache = load(tmp_path, report).analytics["cache"][0]
    assert cache == dict(t=0.9, pairs=4, hits=2, misses=2, hitRate=0.5, missRate=0.5,
                         falseHits=1, falseMisses=0, falseHitRate=1/3, falseMissRate=0, seeded=3, lookups=12, avoided=6, lookup=0)


def test_escalated_attempt_need_not_supply_returned_answer(tmp_path, report):
    metrics = load(tmp_path, report).analytics
    assert (metrics["small"], metrics["large"]) == (2, 0)
    assert metrics["attempts"] == {"small": 2, "large": 1}
    assert metrics["final"] == {"small": 2, "large": 0}
    assert (metrics["escalationRate"], metrics["acceptanceRate"], metrics["successRate"]) == (0.5, 0.5, 0)
    report["pipeline_benchmark"]["cases"][0]["final_attempt_index"] = 1
    assert load(tmp_path, report).analytics["final"] == {"small": 1, "large": 1}


def test_missing_is_local_and_distinct_from_zero(tmp_path, report):
    cache = report["cache_benchmarks"][0]
    del cache["metrics"]["false_hits"]
    cache["rows"].pop()
    del cache["reuse"]["generation_calls_avoided"]
    del report["pipeline_benchmark"]["routing_metrics"]["all_attempt_roles"]["large"]
    metrics = load(tmp_path, report).analytics
    c = metrics["cache"][0]
    assert c["hits"] is c["hitRate"] is c["falseHits"] is c["avoided"] is c["seeded"] is None
    assert c["missRate"] == 0.5 and c["falseMisses"] == c["lookup"] == 0
    assert metrics["attempts"] == {"small": 2, "large": None}
    assert metrics["passed"] is None


@pytest.mark.parametrize("index", [-1, 2, True, None, "0"])
def test_unusable_final_index_never_infers_role(tmp_path, report, index):
    report["pipeline_benchmark"]["cases"][0]["final_attempt_index"] = index
    metrics = load(tmp_path, report).analytics
    assert metrics["final"] == {"small": None, "large": None}
    assert metrics["escalations"] == 1


def test_partial_case_coverage_never_reported_as_full_distribution(tmp_path, report):
    report["pipeline_benchmark"]["cases"].pop()
    assert load(tmp_path, report).analytics["final"] == {"small": None, "large": None}


@pytest.mark.parametrize(("path", "value"), [
    (("cache_benchmarks", 0, "metrics", "true_hits"), True),
    (("cache_benchmarks", 0, "metrics", "false_hits"), -1),
    (("cache_benchmarks", 0, "metrics", "true_misses"), 1.5),
    (("cache_benchmarks", 0, "metrics", "false_misses"), 2),
    (("cache_benchmarks", 0, "reuse", "generation_calls_avoided"), 15),
    (("cache_benchmarks", 0, "latency", "average_lookup_ms"), "12"),
    (("cache_benchmarks", 0, "rows", 1, "id"), "0"),
    (("pipeline_benchmark", "routing_metrics", "initial_selections", "small"), 0),
    (("pipeline_benchmark", "routing_metrics", "all_attempt_roles", "large"), 2),
    (("pipeline_benchmark", "escalation_metrics", "successful_escalations"), 2),
])
def test_malformed_and_contradictory_report_is_unavailable(tmp_path, report, path, value):
    node = report
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    result = load(tmp_path, report)
    assert result.status == "invalid" and result.analytics is None


def test_zero_denominators_preserve_observed_zero_counts(tmp_path, report):
    report["datasets"].update(pair_count=0, routing_case_count=0)
    report["cache_benchmarks"][0].update(pair_count=0, rows=[], metrics=dict.fromkeys(
        ["true_hits", "false_hits", "true_misses", "false_misses"], 0), reuse={"lookup_count": 0})
    report["pipeline_benchmark"] = {"routing_metrics": {"query_count": 0}, "cases": [],
                                    "escalation_metrics": {"escalations_triggered": 0,
                                                           "successful_escalations": 0}}
    result = load(tmp_path, report)
    assert result.status == "empty"
    b = result.analytics
    cache = b["cache"][0]
    assert cache["hits"] == cache["misses"] == cache["seeded"] == 0
    assert cache["hitRate"] is cache["missRate"] is cache["lookup"] is None
    assert b["escalationRate"] is b["acceptanceRate"] is b["successRate"] is None
    assert b["final"] == {"small": 0, "large": 0}


def test_no_triggered_escalations_with_nonempty_queries(tmp_path, report):
    pipeline = report["pipeline_benchmark"]
    pipeline["escalation_metrics"].update(escalations_triggered=0, total_model_attempts=2)
    pipeline["routing_metrics"]["all_attempt_roles"]["large"] = 0
    pipeline["cases"][0].update(escalated=False, attempts=[{"role": "small"}])
    b = load(tmp_path, report).analytics
    assert b["successRate"] is None and b["escalationRate"] == 0


def test_historical_only_counts_role_tagged_requests_including_failures(tmp_path, report):
    synthetic = load(tmp_path, report)
    assert load_runtime(tmp_path).status == "missing"
    row = dict(timestamp_utc="2026-01-01T00:00:00Z", model="gemini-test",
               latency_ms=None, input_tokens=None, output_tokens=None,
               estimated_cost_usd=None, cost_status="usage_unavailable")
    kinds = ["large_model", "large_model", "answer", "embedding", "memory_summary"]
    rows = [{**row, "request_kind": k, "status": "error" if i == 0 else "success"}
            for i, k in enumerate(kinds)]
    (tmp_path / "request_metrics.jsonl").write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
    historical = load_runtime(tmp_path)
    assert synthetic.label != historical.label
    assert (historical.metadata.small_attempts, historical.metadata.large_attempts) == (0, 2)
    assert load_benchmark(tmp_path).analytics == synthetic.analytics
