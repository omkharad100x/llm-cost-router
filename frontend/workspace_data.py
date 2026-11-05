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


def number(obj, key):
    n = obj.get(key)
    if type(n) not in (int, float) or not math.isfinite(n) or n < 0:
        raise ValueError()
    return n


def benchmark():
    d = ds._json(ds._read_source(ds.PROJECT_ROOT, "benchmark"))
    ds.validate_benchmark_report(d)
    cache = []
    for c in d["cache_benchmarks"]:
        m = c["metrics"]
        cache.append(dict(t=c["threshold"], hits=number(m,"true_hits")+number(m,"false_hits"),
                          misses=number(m,"true_misses")+number(m,"false_misses"),
                          falseHits=number(m,"false_hits"), falseMisses=number(m,"false_misses"),
                          avoided=number(c["reuse"],"generation_calls_avoided"),
                          lookup=number(c["latency"],"average_lookup_ms")))
    p = d["pipeline_benchmark"]
    routing, esc = p["routing_metrics"], p["escalation_metrics"]
    counts = {k:number(routing["initial_selections"],k) for k in ("small","large")}
    final = dict(small=0, large=0)
    for case in p["cases"]:
        role = "large" if case["escalated"] else case["initial_role"]
        final[role] += 1
    return dict(cache=cache, **counts, final=final, simple=number(routing["classifications"],"simple"),
                complex=number(routing["classifications"],"complex"), queries=number(routing,"query_count"),
                escalations=number(esc,"escalations_triggered"), successes=number(esc,"successful_escalations"),
                accepted=number(esc,"initial_answers_accepted"), passed=number(p["quality_metrics"],"final_passes"))


def read(fn):
    try:
        return fn(), "Ready"
    except ds._SourceIssue as e:
        return None, e.status.capitalize()
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError):
        return None, "Invalid"


history, hs = read(historical)
bench, bs = read(benchmark)
models = load_model_config()
print(json.dumps(dict(history=history, hs=hs, bench=bench, bs=bs, small=models.small_model,
                      large=models.large_model, schema=SCHEMA_VERSION, version=DEFAULT_CONFIG.cache_version,
                      threshold=DEFAULT_CONFIG.similarity_threshold, evaluation=EvaluationConfig().acceptance_threshold)))
