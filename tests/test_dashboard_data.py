"""Offline source validation and safe projection tests; no private files."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

import dashboard_data as data


SECRET = "synthetic-sensitive-marker"


def benchmark():
    return {
        "report_schema_version": 1,
        "datasets": {"pair_count": 2, "routing_case_count": 1},
        "configuration": {"thresholds": [0.8, 0.9]},
        "cache_benchmarks": [{"threshold": 0.8, "pair_count": 2},
                             {"threshold": 0.9, "pair_count": 2}],
        "pipeline_benchmark": {"routing_metrics": {"query_count": 1}},
    }


def runtime():
    return {
        "timestamp_utc": "2026-11-05T00:00:00+00:00",
        "model": "gemini-test", "request_kind": "answer", "status": "success",
        "latency_ms": 12.5, "input_tokens": 10, "output_tokens": 2,
        "estimated_cost_usd": "0.000001", "cost_status": "estimated",
    }


def write_report(directory, payload):
    (directory / "benchmark_report.json").write_text(json.dumps(payload), encoding="utf-8")


def write_runtime(directory, rows):
    (directory / "request_metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def deny_private_file_access(monkeypatch):
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert path.name not in {".env", "memory.txt", "secrets.toml", "credentials.toml"}
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)


def test_benchmark_metadata_projection_and_label(tmp_path):
    payload = benchmark()
    payload["datasets"]["source"] = SECRET
    payload["query"] = SECRET
    payload["configuration"]["api_key"] = SECRET
    write_report(tmp_path, payload)
    result = data.load_benchmark(tmp_path)
    assert result.status == "ready"
    assert result.metadata == data.BenchmarkMetadata(2, 1, (0.8, 0.9))
    assert "synthetic" in result.label and "Offline" in result.label
    assert SECRET not in repr(result)


def test_actual_existing_benchmark_report_is_supported():
    result = data.load_benchmark()
    assert result.status == "ready"
    assert result.metadata.pair_count == 22 and result.metadata.routing_query_count == 8


@pytest.mark.parametrize("loader", [data.load_benchmark, data.load_runtime])
def test_missing_source(loader, tmp_path):
    result = loader(tmp_path)
    assert result.status == "missing" and result.metadata is None


@pytest.mark.parametrize("filename,loader", [
    ("benchmark_report.json", data.load_benchmark), ("request_metrics.jsonl", data.load_runtime),
])
@pytest.mark.parametrize("text", ["", " \n\n"])
def test_blank_source(filename, loader, text, tmp_path):
    (tmp_path / filename).write_text(text, encoding="utf-8")
    assert loader(tmp_path).status == "empty"


@pytest.mark.parametrize("filename,loader", [
    ("benchmark_report.json", data.load_benchmark), ("request_metrics.jsonl", data.load_runtime),
])
@pytest.mark.parametrize("text", ['{"secret": "synthetic-sensitive-marker", broken', "[]", "null", "NaN", "1e999"])
def test_malformed_sources_are_generic_and_never_echo_input(filename, loader, text, tmp_path):
    (tmp_path / filename).write_text(text, encoding="utf-8")
    result = loader(tmp_path)
    assert result.status == "invalid" and result.metadata is None
    assert SECRET not in repr(result) and str(tmp_path) not in repr(result)


def test_empty_benchmark_datasets_are_not_live_activity(tmp_path):
    payload = benchmark()
    payload["datasets"] = {"pair_count": 0, "routing_case_count": 0}
    for experiment in payload["cache_benchmarks"]:
        experiment["pair_count"] = 0
    payload["pipeline_benchmark"]["routing_metrics"]["query_count"] = 0
    write_report(tmp_path, payload)
    result = data.load_benchmark(tmp_path)
    assert result.status == "empty"
    assert result.metadata.pair_count == 0
    assert "synthetic" in result.label


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(report_schema_version=2),
    lambda p: p.update(report_schema_version=True),
    lambda p: p.update(datasets=None),
    lambda p: p["datasets"].update(pair_count=-1),
    lambda p: p["datasets"].update(routing_case_count=True),
    lambda p: p["configuration"].update(thresholds=[]),
    lambda p: p["configuration"].update(thresholds=[0.9, 0.9]),
    lambda p: p["configuration"].update(thresholds=[True]),
    lambda p: p["configuration"].update(thresholds=[1.1]),
    lambda p: p["configuration"].update(thresholds=[float("inf")]),
    lambda p: p["cache_benchmarks"][0].update(pair_count=99),
    lambda p: p["cache_benchmarks"][0].update(threshold=True),
    lambda p: p["cache_benchmarks"].pop(),
    lambda p: p["pipeline_benchmark"]["routing_metrics"].update(query_count=5),
])
def test_invalid_benchmark_metadata(mutate, tmp_path):
    payload = benchmark()
    mutate(payload)
    write_report(tmp_path, payload)
    assert data.load_benchmark(tmp_path).status == "invalid"


def test_duplicate_json_keys_rejected(tmp_path):
    (tmp_path / "benchmark_report.json").write_text(
        '{"report_schema_version": 1, "report_schema_version": 2}', encoding="utf-8",
    )
    assert data.load_benchmark(tmp_path).status == "invalid"


def test_runtime_projection_latest_timestamp_and_distinct_label(tmp_path):
    first = runtime()
    second = {**first, "timestamp_utc": "2026-11-05T06:30:00+05:30", "query": SECRET,
              "response": SECRET, "api_key": SECRET}
    write_runtime(tmp_path, [second, first])
    result = data.load_runtime(tmp_path)
    assert result.metadata == data.RuntimeMetadata(2, "2026-11-05T01:00:00+00:00")
    assert result.status == "ready" and result.label == "Local request records"
    assert SECRET not in repr(result) and "live" not in result.label.lower()


@pytest.mark.parametrize("value", [None, 0])
def test_runtime_missing_usage_and_zero_are_both_valid(tmp_path, value):
    row = runtime()
    row.update(input_tokens=value, output_tokens=value,
               estimated_cost_usd=None if value is None else "0",
               cost_status="usage_unavailable" if value is None else "estimated")
    write_runtime(tmp_path, [row])
    assert data.load_runtime(tmp_path).status == "ready"


@pytest.mark.parametrize("kind", sorted(data.REQUEST_KINDS))
def test_existing_instrumentation_request_kinds(kind, tmp_path):
    row = {**runtime(), "request_kind": kind}
    write_runtime(tmp_path, [row])
    assert data.load_runtime(tmp_path).status == "ready"


@pytest.mark.parametrize("field,value", [
    ("timestamp_utc", SECRET), ("timestamp_utc", "2026-11-05T00:00:00"),
    ("status", SECRET), ("request_kind", SECRET), ("model", SECRET),
    ("latency_ms", -1), ("latency_ms", True), ("latency_ms", float("nan")),
    ("input_tokens", -1), ("output_tokens", True), ("cost_status", SECRET),
    ("estimated_cost_usd", "-1"), ("estimated_cost_usd", "NaN"),
    ("estimated_cost_usd", SECRET), ("estimated_cost_usd", None),
    ("cost_status", "pricing_unavailable"),
])
def test_invalid_runtime_records_fail_without_partial_results(field, value, tmp_path):
    write_runtime(tmp_path, [runtime(), {**runtime(), field: value}])
    result = data.load_runtime(tmp_path)
    assert result.status == "invalid" and result.metadata is None
    assert SECRET not in repr(result)


def test_partial_jsonl_line_is_rejected_until_repaired(tmp_path):
    path = tmp_path / "request_metrics.jsonl"
    path.write_text(json.dumps(runtime()) + "\n{", encoding="utf-8")
    assert data.load_runtime(tmp_path).status == "invalid"
    path.write_text(json.dumps(runtime()) + "\n", encoding="utf-8")
    assert data.load_runtime(tmp_path).status == "ready"


@pytest.mark.parametrize("filename,loader", [
    ("benchmark_report.json", data.load_benchmark), ("request_metrics.jsonl", data.load_runtime),
])
def test_size_limits_and_encoding(filename, loader, tmp_path, monkeypatch):
    path = tmp_path / filename
    path.write_bytes(b"x" * 30)
    monkeypatch.setattr(data, "MAX_SOURCE_BYTES", 20)
    assert loader(tmp_path).status == "too_large"
    path.write_bytes(b"\xff\xfe\x00")
    assert loader(tmp_path).status == "invalid"


def test_runtime_record_limit(tmp_path, monkeypatch):
    write_runtime(tmp_path, [runtime(), runtime()])
    monkeypatch.setattr(data, "MAX_RUNTIME_RECORDS", 1)
    assert data.load_runtime(tmp_path).status == "too_large"


@pytest.mark.parametrize("loader", [data.load_benchmark, data.load_runtime])
def test_permission_errors_do_not_expose_raw_exception(loader, tmp_path, monkeypatch):
    write_report(tmp_path, benchmark())
    write_runtime(tmp_path, [runtime()])
    def denied(*args, **kwargs):
        raise PermissionError(SECRET)
    monkeypatch.setattr(Path, "open", denied)
    result = loader(tmp_path)
    assert result.status == "unreadable" and SECRET not in repr(result)


def test_linked_source_is_rejected_before_reading_target(tmp_path, monkeypatch):
    # Simulate the platform's link check without needing Windows symlink privileges.
    def linked(path):
        return path.name == "benchmark_report.json"
    def forbidden(*args, **kwargs):
        raise AssertionError("A linked source must not be opened.")
    monkeypatch.setattr(Path, "is_symlink", linked)
    monkeypatch.setattr(Path, "open", forbidden)
    assert data.load_benchmark(tmp_path).status == "unreadable"
