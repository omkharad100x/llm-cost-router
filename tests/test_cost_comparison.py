"""Paired offline cost scenarios with hand-calculated totals and blocked sockets."""

from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

from benchmarks import BenchmarkConfig, DEFAULT_FIXTURES, benchmark_cost_comparison, load_fixtures
from cost_comparison import comparison_metrics, project_comparison
from dashboard_data import load_benchmark
from evaluate_similarity import load_dataset
from instrumentation import usage_metrics, embedding_usage_metrics
from model_config import ModelConfig


MODELS = {"small": "gemini-small", "large": "gemini-large", "embedding": "gemini-embedding-001"}
# Deliberately synthetic test prices. Production model_pricing.json is never changed.
RATES = {"small": {"input_usd_per_million": "0.25", "output_usd_per_million": "1.50"},
         "large": {"input_usd_per_million": "1", "output_usd_per_million": "2"},
         "embedding": {"input_usd_per_million": "0.15", "output_usd_per_million": "0"}}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network access is forbidden.")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)


def call(role, phase, *, priced=True, usage=True):
    rates = RATES[role] if priced else None
    if role == "embedding":
        response = SimpleNamespace(embeddings=[SimpleNamespace(
            statistics=SimpleNamespace(token_count=10) if usage else None)])
        metrics = embedding_usage_metrics(response, rates)
    else:
        response = SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count=100, cached_content_token_count=0, tool_use_prompt_token_count=0,
            candidates_token_count=20, thoughts_token_count=5, total_token_count=125) if usage else None)
        metrics = usage_metrics(response, rates)
    return dict(model=MODELS[role], request_kind="embedding" if role=="embedding" else role+"_model",
                phase=phase, status="success", **metrics)


@pytest.fixture
def ledger():
    return {"schema_version": 1, "methodology": "paired-routing-fixtures-v1",
            "models": MODELS.copy(), "threshold": 0.9, "queries": [
                {"id": "1", "cache_outcome": "miss", "baseline": [call("large", "baseline")],
                 "optimized": [call("embedding", "embedding"), call("small", "initial"), call("embedding", "embedding")]},
                {"id": "2", "cache_outcome": "safe_hit", "baseline": [call("large", "baseline")],
                 "optimized": [call("embedding", "embedding")]},
                {"id": "3", "cache_outcome": "miss", "baseline": [call("large", "baseline")],
                 "optimized": [call("embedding", "embedding"), call("small", "initial"), call("large", "escalation")]},
            ]}


def test_complete_costs_savings_and_per_query_denominators(ledger):
    result = comparison_metrics(ledger)
    assert Decimal(result["baseline"]["estimated_cost_usd"]) == Decimal("0.00045")
    assert Decimal(result["optimized"]["estimated_cost_usd"]) == Decimal("0.000281")
    assert Decimal(result["savings_usd"]) == Decimal("0.000169")
    assert Decimal(result["savings_percentage"]) == Decimal("0.000169") * 100 / Decimal("0.00045")
    assert Decimal(result["baseline"]["average_cost_per_query_usd"]) == Decimal("0.00015")
    assert Decimal(result["optimized"]["average_cost_per_query_usd"]) == Decimal("0.000281") / 3
    assert result["baseline"]["generation_calls"] == result["optimized"]["generation_calls"] == 3
    assert result["safe_calls_avoided"] == 1  # Escalation offsets the avoided initial call.


def test_multiple_models_escalation_and_paid_cache_lookup(ledger):
    result = comparison_metrics(ledger)
    costs = {r["model"]: r["optimized"] for r in result["by_model"]}
    assert Decimal(costs[MODELS["small"]]["estimated_cost_usd"]) == Decimal("0.000125")
    assert Decimal(result["escalation"]["estimated_cost_usd"]) == Decimal("0.00015")
    assert Decimal(result["embedding"]["estimated_cost_usd"]) == Decimal("0.000006")
    assert result["embedding"]["requests"] == 4
    assert costs[MODELS["large"]]["generation_calls"] == 1


@pytest.mark.parametrize("role", ["small", "large", "embedding"])
@pytest.mark.parametrize("missing", ["pricing", "usage"])
def test_missing_prices_or_tokens_withhold_totals_and_preserve_subtotals(ledger, role, missing):
    for query in ledger["queries"]:
        for scenario in ("baseline", "optimized"):
            for i, record in enumerate(query[scenario]):
                if record["model"] == MODELS[role]:
                    query[scenario][i] = call(role, record["phase"], priced=missing!="pricing", usage=missing!="usage")
    result = comparison_metrics(ledger)
    assert result["savings_usd"] is result["savings_percentage"] is None
    assert result["optimized"]["estimated_cost_usd"] is None
    expected = {"small": "0.000156", "large": "0.000131", "embedding": "0.000275"}[role]
    assert Decimal(result["optimized"]["known_cost_subtotal_usd"]) == Decimal(expected)
    assert any(MODELS[role] in issue for issue in result["issues"])


@pytest.mark.parametrize("outcome", ["false_hit", "unreviewed_hit"])
def test_unsafe_cache_hits_never_become_savings(ledger, outcome):
    ledger["queries"][1]["cache_outcome"] = outcome
    result = comparison_metrics(ledger)
    assert result["baseline"]["complete"] and result["optimized"]["complete"]
    assert result["safe_calls_avoided"] == 0
    assert result["savings_usd"] is result["savings_percentage"] is None
    assert result["cache_outcomes"][outcome] == 1


def test_miss_requires_paid_generation_and_can_increase_spend(ledger):
    ledger["queries"] = [ledger["queries"][2]]
    result = comparison_metrics(ledger)
    assert result["safe_calls_avoided"] == 0
    assert Decimal(result["savings_usd"]) < 0
    assert result["optimized"]["generation_calls"] == 2


def test_empty_workload_and_zero_baseline(ledger):
    ledger["queries"] = []
    result = comparison_metrics(ledger)
    assert result["baseline"]["estimated_cost_usd"] == result["optimized"]["estimated_cost_usd"] == "0"
    assert result["baseline"]["average_cost_per_query_usd"] is None
    assert result["savings_percentage"] is None


def test_zero_price_is_distinct_from_unavailable(ledger):
    for query in ledger["queries"]:
        for record in query["baseline"] + query["optimized"]:
            record["estimated_cost_usd"] = "0"
            for field in ("input_cost_usd", "cached_input_cost_usd", "output_cost_usd"):
                record.pop(field, None)
    result = comparison_metrics(ledger)
    assert result["savings_usd"] == "0"
    assert result["savings_percentage"] is None
    assert result["baseline"]["average_cost_per_query_usd"] == "0"


def test_decimal_precision_retained_through_json(ledger):
    for query in ledger["queries"]:
        query["baseline"][0]["estimated_cost_usd"] = "0.100000000000000000000000001"
        for field in ("input_cost_usd", "cached_input_cost_usd", "output_cost_usd"):
            query["baseline"][0].pop(field, None)
    result = comparison_metrics(json.loads(json.dumps(ledger)))
    assert result["baseline"]["estimated_cost_usd"] == "0.300000000000000000000000003"
    assert Decimal(result["savings_usd"]) == Decimal("0.299719000000000000000000003")


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(schema_version=True),
    lambda p: p.update(threshold=float("nan")),
    lambda p: p.update(threshold=2),
    lambda p: p.update(queries=None),
    lambda p: p["queries"].append(deepcopy(p["queries"][0])),
    lambda p: p["queries"][0].update(baseline=[]),
    lambda p: p["queries"][0].update(optimized=[]),
    lambda p: p["queries"][0].update(cache_outcome="safe_hit"),
    lambda p: p["queries"][1].update(optimized=[]),
    lambda p: p["models"].update(large="sensitive-string"),
    lambda p: p["queries"][0]["baseline"][0].update(phase="initial"),
    lambda p: p["queries"][0]["baseline"][0].update(cost_status="usage_unavailable"),
    lambda p: p["queries"][0]["baseline"][0].update(estimated_cost_usd="NaN"),
    lambda p: p["queries"][0]["baseline"][0].update(estimated_cost_usd="-1"),
    lambda p: p["queries"][0]["baseline"][0].update(estimated_cost_usd=0.1),
    lambda p: p["queries"][0]["baseline"][0].update(input_tokens=True),
    lambda p: p["queries"][0]["baseline"][0].update(billed_output_tokens=None),
])
def test_invalid_ledgers_are_unavailable_with_safe_errors(ledger, mutate):
    mutate(ledger)
    result = project_comparison(ledger)
    assert result["status"] == "invalid" and "sensitive-string" not in repr(result)


def test_old_reports_and_historical_data_are_separate(tmp_path, ledger):
    report = json.loads((Path(__file__).resolve().parents[1] / "benchmark_report.json").read_bytes())
    report["cost_comparison"] = ledger
    target = tmp_path / "benchmark_report.json"
    target.write_text(json.dumps(report), encoding="utf-8")
    first = load_benchmark(tmp_path).analytics["comparison"]
    (tmp_path / "request_metrics.jsonl").write_text("unrelated history", encoding="utf-8")
    assert load_benchmark(tmp_path).analytics["comparison"] == first
    report.pop("cost_comparison")
    target.write_text(json.dumps(report), encoding="utf-8")
    assert load_benchmark(tmp_path).analytics["comparison"]["status"] == "missing"
    report["cost_comparison"] = {}
    target.write_text(json.dumps(report), encoding="utf-8")
    result = load_benchmark(tmp_path)
    assert result.status == "ready" and result.analytics["comparison"]["status"] == "invalid"


def test_real_pipeline_uses_reviewed_cache_and_existing_instrumentation(tmp_path):
    fixtures = load_fixtures(DEFAULT_FIXTURES, load_dataset())
    models = ModelConfig()
    pricing = {getattr(models, role+"_model"): RATES[role] for role in ("small","large")}
    pricing[MODELS["embedding"]] = RATES["embedding"]
    fixtures["embedding_token_count"] = 10
    # Keep the existing missing-usage fixture missing.
    ledger = benchmark_cost_comparison(fixtures, pricing, BenchmarkConfig())
    result = comparison_metrics(ledger)
    assert result["query_count"] == 16
    assert result["safe_calls_avoided"] == 6
    assert result["baseline"]["generation_calls"] == 16
    assert result["optimized"]["generation_calls"] == 15
    assert result["embedding"]["requests"] == 20
    assert result["escalation"]["requests"] == 5
    assert result["optimized"]["estimated_cost_usd"] is None
    assert Decimal(result["embedding"]["estimated_cost_usd"]) == Decimal("0.00003")
    # Force unrelated reviewed queries to collide, exercising actual similarity lookup.
    fixtures["comparison_cache_reviews"]["escalation_pass"]["embedding"] = fixtures["comparison_cache_reviews"]["simple_pass"]["embedding"]
    unsafe = comparison_metrics(benchmark_cost_comparison(fixtures, pricing, BenchmarkConfig()))
    assert unsafe["cache_outcomes"]["false_hit"] == 2
    assert unsafe["savings_usd"] is None
