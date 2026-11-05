"""Bounded, read-only dashboard sources; no clients, environment, or UI imports."""

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path
import re

from dashboard_metrics import benchmark_metrics


PROJECT_ROOT = Path(__file__).resolve().parent
MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_RUNTIME_RECORDS = 10000
SOURCE_FILES = {"benchmark": "benchmark_report.json", "runtime": "request_metrics.jsonl"}
SOURCE_LABELS = {"benchmark": "Offline benchmark · synthetic", "runtime": "Local request records"}
COST_STATUSES = {
    "estimated", "pricing_unavailable", "usage_unavailable", "usage_inconsistent",
    "cached_pricing_unavailable", "metrics_unavailable",
}
REQUEST_KINDS = {"answer", "memory_summary", "embedding", "small_model", "large_model"}


@dataclass(frozen=True)
class BenchmarkMetadata:
    pair_count: int
    routing_query_count: int
    thresholds: tuple[float, ...]
    hit_counts: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class RuntimeMetadata:
    record_count: int
    latest_timestamp_utc: str
    small_attempts: int = 0
    large_attempts: int = 0


@dataclass(frozen=True)
class SourceResult:
    kind: str
    status: str
    message: str
    metadata: BenchmarkMetadata | RuntimeMetadata | None = None
    analytics: dict | None = None

    @property
    def label(self):
        return SOURCE_LABELS[self.kind]


class _SourceIssue(Exception):
    def __init__(self, status):
        self.status = status


def _read_source(directory, kind):
    """Only fixed source names are accepted; linked files are never followed."""
    try:
        root = Path(directory).resolve()
        path = root / SOURCE_FILES[kind]
        if path.is_symlink() or path.resolve() != path:
            raise _SourceIssue("unreadable")
        if path.stat().st_size > MAX_SOURCE_BYTES:
            raise _SourceIssue("too_large")
        with path.open("rb") as file:
            content = file.read(MAX_SOURCE_BYTES + 1)
        if len(content) > MAX_SOURCE_BYTES:
            raise _SourceIssue("too_large")
        return content.decode("utf-8")
    except FileNotFoundError:
        raise _SourceIssue("missing") from None
    except UnicodeError:
        raise _SourceIssue("invalid") from None
    except OSError:
        raise _SourceIssue("unreadable") from None


def _reject_constant(value):
    raise ValueError("Invalid JSON number.")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Invalid JSON number.")
    return number


def _unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate JSON field.")
        result[key] = value
    return result


def _json(text):
    return json.loads(
        text, parse_constant=_reject_constant, parse_float=_finite_float,
        object_pairs_hook=_unique_object,
    )


def _count(value):
    if type(value) is not int or value < 0:
        raise ValueError("Expected a nonnegative integer.")
    return value


def validate_benchmark_report(payload):
    """Validate the report envelope before projecting dashboard metrics.

    Return only typed counts and thresholds. Text, configuration secrets, raw
    responses, and unknown fields are never passed through to the dashboard.
    """
    if not isinstance(payload, dict) or type(payload.get("report_schema_version")) is not int:
        raise ValueError("Expected a versioned benchmark report.")
    if payload["report_schema_version"] != 1:
        raise ValueError("Unsupported benchmark report version.")
    datasets = payload["datasets"]
    pairs = _count(datasets["pair_count"])
    queries = _count(datasets["routing_case_count"])
    thresholds = payload["configuration"]["thresholds"]
    if not isinstance(thresholds, list) or not thresholds:
        raise ValueError("Expected benchmark thresholds.")
    for threshold in thresholds:
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not -1 <= threshold <= 1:
            raise ValueError("Invalid benchmark threshold.")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("Duplicate benchmark threshold.")
    experiments = payload["cache_benchmarks"]
    if not isinstance(experiments, list) or len(experiments) != len(thresholds):
        raise ValueError("Inconsistent benchmark experiments.")
    counts = []
    for experiment, threshold in zip(experiments, thresholds):
        if (
            not isinstance(experiment, dict)
            or type(experiment.get("threshold")) not in (int, float)
            or experiment["threshold"] != threshold
            or _count(experiment["pair_count"]) != pairs
        ):
            raise ValueError("Inconsistent benchmark metadata.")
        metrics = experiment.get("metrics")
        if isinstance(metrics, dict):
            hits = (metrics.get("true_hits"), metrics.get("false_hits"))
            if all(type(n) is int and n >= 0 for n in hits) and sum(hits) <= pairs:
                counts.append(hits)
    if _count(payload["pipeline_benchmark"]["routing_metrics"]["query_count"]) != queries:
        raise ValueError("Inconsistent routing dataset size.")
    return BenchmarkMetadata(pairs, queries, tuple(float(value) for value in thresholds),
                             tuple(counts) if len(counts) == len(thresholds) else ())


def validate_runtime_record(record):
    """Check the existing instrumentation schema; return only a UTC timestamp."""
    if not isinstance(record, dict):
        raise ValueError("Expected a metrics object.")
    if record["status"] not in ("success", "error") or record["request_kind"] not in REQUEST_KINDS:
        raise ValueError("Unknown request metadata.")
    if not isinstance(record["model"], str) or not re.fullmatch(r"gemini-[A-Za-z0-9._-]{1,120}", record["model"]):
        raise ValueError("Invalid model metadata.")
    if not isinstance(record["timestamp_utc"], str):
        raise ValueError("Invalid timestamp.")
    timestamp = datetime.fromisoformat(record["timestamp_utc"])
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("Timestamp requires a timezone.")
    for field in ("input_tokens", "output_tokens"):
        if field not in record:
            raise ValueError("Missing usage field.")
    for field in (
        "input_tokens", "output_tokens", "thinking_tokens", "billed_output_tokens",
        "total_tokens", "cached_input_tokens",
    ):
        value = record.get(field)
        if value is not None:
            _count(value)
    latency = record["latency_ms"]
    if latency is not None and (
        type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0
    ):
        raise ValueError("Invalid latency.")
    if record["cost_status"] not in COST_STATUSES:
        raise ValueError("Unknown cost status.")
    cost = record["estimated_cost_usd"]
    if cost is not None:
        if not isinstance(cost, str) or len(cost) > 64:
            raise ValueError("Invalid cost.")
        try:
            amount = Decimal(cost)
            if not amount.is_finite() or amount < 0:
                raise ValueError("Invalid cost.")
        except InvalidOperation:
            raise ValueError("Invalid cost.") from None
    if (record["cost_status"] == "estimated") != (cost is not None):
        raise ValueError("Inconsistent cost availability.")
    return timestamp.astimezone(timezone.utc)


def _result(kind, status, metadata=None, analytics=None):
    messages = {
        "ready": "Source metadata validated. Ready for the analytics workspace.",
        "empty": "No observations are available in this source yet.",
        "missing": "No source file found. Add the expected local report, then refresh.",
        "invalid": "This source could not be validated. Check its JSON and supported format, then refresh.",
        "unreadable": "This source could not be read safely. Check file access and use a regular local file.",
        "too_large": "This source exceeds the dashboard limit. Use a smaller report or rotate the request log.",
    }
    return SourceResult(kind, status, messages[status], metadata, analytics)


def load_benchmark(directory=PROJECT_ROOT):
    try:
        text = _read_source(directory, "benchmark")
        if not text.strip():
            return _result("benchmark", "empty")
        report = _json(text)
        metadata = validate_benchmark_report(report)
        status = "ready" if metadata.pair_count or metadata.routing_query_count else "empty"
        return _result("benchmark", status, metadata, benchmark_metrics(report))
    except _SourceIssue as error:
        return _result("benchmark", error.status)
    except (ValueError, TypeError, KeyError, IndexError, OverflowError, RecursionError):
        return _result("benchmark", "invalid")


def load_runtime(directory=PROJECT_ROOT):
    try:
        text = _read_source(directory, "runtime")
        count = 0
        latest = None
        roles = dict(small_model=0, large_model=0)
        for line in text.splitlines():
            if not line.strip():
                continue
            count += 1
            if count > MAX_RUNTIME_RECORDS:
                return _result("runtime", "too_large")
            record = _json(line)
            timestamp = validate_runtime_record(record)
            if record['request_kind'] in roles:
                roles[record['request_kind']] += 1
            latest = timestamp if latest is None else max(latest, timestamp)
        if not count:
            return _result("runtime", "empty")
        metadata = RuntimeMetadata(count, latest.isoformat(), roles["small_model"], roles["large_model"])
        return _result("runtime", "ready", metadata)
    except _SourceIssue as error:
        return _result("runtime", error.status)
    except (ValueError, TypeError, KeyError, IndexError, OverflowError, RecursionError):
        return _result("runtime", "invalid")
