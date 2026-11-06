"""Offline latency, quality and paired dashboard projections."""
import copy
import pytest
from latency_quality import latency_stats, quality_stats, project_latency_quality
from benchmarks import run_benchmarks
from dashboard_metrics import benchmark_metrics

def test_percentiles_and_zero_duration():
    x=latency_stats([0,10,20,30,40])
    assert (x["average_ms"],x["p50_ms"],x["p95_ms"],x["p99_ms"],x["min_ms"],x["max_ms"]) == (20,20,40,40,0,40)
    assert latency_stats([])["average_ms"] is None

@pytest.mark.parametrize("bad",[-1,float("nan"),float("inf"),None,True])
def test_invalid_measurement(bad):
    with pytest.raises(ValueError):
        latency_stats([bad])

def test_quality_counts_and_missing():
    assert quality_stats(["pass","fail","pass"]) == dict(total=3,passed=2,failed=1,pass_rate=2/3)
    assert quality_stats([])["pass_rate"] is None
    with pytest.raises(ValueError):
        quality_stats(["unknown"])

def test_paired_benchmark_quality_and_components():
    report=run_benchmarks()
    result=benchmark_metrics(report)["performance"]
    assert result["baseline"]["count"] == result["optimized"]["count"] == 16
    assert result["cache_hit"]["count"] == 6
    assert result["cache_lookup"]["count"] == 16
    assert result["embedding"]["count"] == 16  # per-query samples; 20 recorded embedding calls
    assert result["small_generation"]["count"] > 0
    assert result["large_generation"]["count"] > 0
    assert result["escalation"]["count"] == result["escalation_count"] == 5
    assert result["baseline_quality"]["total"] == result["optimized_quality"]["total"] == 16
    assert result["initial_quality"]["total"] == 10
    assert not result["source"].startswith("live")

def test_old_and_incomplete_ledgers_remain_unavailable():
    assert project_latency_quality(None) is None
    assert project_latency_quality({"queries":[{"id":"old"}]}) is None
    report=run_benchmarks()
    ledger=copy.deepcopy(report["cost_comparison"])
    ledger["queries"][0]["timing"]["optimized"]["end_to_end_ms"]=None
    result=project_latency_quality(ledger)
    assert result["optimized"]["average_ms"] is None
    assert result["improvement_ms"] is None
    ledger=copy.deepcopy(report["cost_comparison"])
    ledger["queries"][0]["cache_outcome"]="false_hit"
    result=project_latency_quality(ledger)
    assert result["improvement_ms"] is None
