"""Read-only projection for the local dashboard. Never return prompts or raw records."""
import json
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dashboard_data as ds
from cache_config import DEFAULT_CONFIG
from cache_schema import SCHEMA_VERSION
from evaluation_config import EvaluationConfig
from model_config import load_model_config


def historical():
    rows = []
    for line in ds._read_source(ds.PROJECT_ROOT, "runtime").splitlines():
        if not line.strip():
            continue
        if len(rows) >= ds.MAX_RUNTIME_RECORDS:
            raise ValueError()
        r = ds._json(line)
        stamp = ds.validate_runtime_record(r)
        cost = r["estimated_cost_usd"]
        if cost is not None and not math.isfinite(float(cost)):
            raise ValueError()
        rows.append(dict(time=stamp.isoformat(), model=r["model"], kind=r["request_kind"],
                         status=r["status"], ms=r["latency_ms"], cost=float(cost) if cost is not None else None,
                         input=r["input_tokens"], output=r["output_tokens"]))
    return rows


def read(fn):
    try:
        return fn(), "Ready"
    except ds._SourceIssue as e:
        return None, e.status.capitalize()
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError):
        return None, "Invalid"


history, hs = read(historical)
report = ds.load_benchmark()
bench, bs = report.analytics, report.status.capitalize()
models = load_model_config()
print(json.dumps(dict(history=history, hs=hs, bench=bench, bs=bs, small=models.small_model,
                      large=models.large_model, schema=SCHEMA_VERSION, version=DEFAULT_CONFIG.cache_version,
                      threshold=DEFAULT_CONFIG.similarity_threshold, evaluation=EvaluationConfig().acceptance_threshold)))
