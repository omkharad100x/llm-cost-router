"""Read-only dashboard projection. Monetary aggregation happens in Python."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dashboard_data as ds
from dashboard_metrics import historical_metrics
from cache_config import DEFAULT_CONFIG
from cache_schema import SCHEMA_VERSION
from evaluation_config import EvaluationConfig
from model_config import load_model_config


def workspace(options=None):
    options = options or {}
    runtime = ds.load_runtime(ds.PROJECT_ROOT)
    report = ds.load_benchmark(ds.PROJECT_ROOT)
    history = costs = None
    if runtime.analytics is not None:
        records = runtime.analytics["records"]
        history = [dict(time=r["timestamp_utc"], model=r["model"], kind=r["request_kind"],
                        status=r["status"], ms=r["latency_ms"], cost=r["estimated_cost_usd"],
                        input=r["input_tokens"], output=r["output_tokens"]) for r in records]
        try:
            costs = historical_metrics(records, start=options.get("from"), end=options.get("to"),
                                       model=options.get("model"), currency=options.get("currency", "USD"),
                                       rate=options.get("rate", "1"))
        except (ValueError, TypeError, ArithmeticError):
            costs = None
    models = load_model_config()
    return dict(history=history, costs=costs,
                hs="Ready" if runtime.status in ("ready", "empty") else runtime.status.capitalize(),
                bench=report.analytics, bs=report.status.capitalize(),
                small=models.small_model, large=models.large_model, schema=SCHEMA_VERSION,
                version=DEFAULT_CONFIG.cache_version, threshold=DEFAULT_CONFIG.similarity_threshold,
                evaluation=EvaluationConfig().acceptance_threshold)


if __name__ == "__main__":
    options = json.loads(sys.argv[1]) if len(sys.argv) == 2 else {}
    print(json.dumps(workspace(options), allow_nan=False))
