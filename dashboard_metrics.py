"""Numeric projections from a validated version-1 offline benchmark envelope."""

import math
from datetime import date

from benchmark_accounting import summarize_requests
from money import amount, multiply, cost_display

from cost_comparison import project_comparison


def _group(obj, key):
    value = obj.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Invalid benchmark group.")
    return value


def _count(obj, key, maximum=None):
    value = obj.get(key)
    if value is None:
        return None
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        raise ValueError("Invalid benchmark count.")
    return value


def _partition(values, total):
    """Allow missing cells, but reject totals that contradict known counts."""
    known = [v for v in values if v is not None]
    if total is not None and (sum(known) > total or
                              (len(known) == len(values) and sum(known) != total)):
        raise ValueError("Inconsistent benchmark totals.")


def _sum_counts(a, b):
    return None if a is None or b is None else a + b


def _rate(count, total):
    return count / total if count is not None and total else None


def _complete_rows(obj, key, size):
    rows = obj.get(key)
    if rows is None:
        return None
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Invalid benchmark observations.")
    ids = [row.get("id") for row in rows]
    if any(not isinstance(i, str) or not i for i in ids):
        return None
    if len(rows) > size or len(set(ids)) != len(ids):
        raise ValueError("Invalid benchmark observation coverage.")
    return rows if len(rows) == size else None


def _cache(experiment):
    pairs = experiment["pair_count"]
    metrics = _group(experiment, "metrics")
    tp, fp, tn, fn = (_count(metrics, k, pairs) for k in
                      ("true_hits", "false_hits", "true_misses", "false_misses"))
    _partition((tp, fp, tn, fn), pairs)
    hits, misses = _sum_counts(tp, fp), _sum_counts(tn, fn)
    reuse = _group(experiment, "reuse")
    lookups = _count(reuse, "lookup_count")
    avoided = _count(reuse, "generation_calls_avoided", lookups)
    _partition((avoided, _count(reuse, "generation_calls")), lookups)
    latency = _group(experiment, "latency").get("average_lookup_ms")
    if latency is not None and (type(latency) not in (int, float) or
                                not math.isfinite(latency) or latency < 0):
        raise ValueError("Invalid lookup latency.")
    rows = _complete_rows(experiment, "rows", pairs)
    seeded = None
    if rows is not None and all(type(row.get("seeded")) is bool for row in rows):
        seeded = sum(row["seeded"] for row in rows)
    return dict(t=experiment["threshold"], pairs=pairs, hits=hits, misses=misses,
                hitRate=_rate(hits, pairs), missRate=_rate(misses, pairs),
                falseHits=fp, falseMisses=fn, falseHitRate=_rate(fp, _sum_counts(fp, tn)),
                falseMissRate=_rate(fn, _sum_counts(fn, tp)), seeded=seeded, lookups=lookups,
                avoided=avoided, lookup=latency if pairs and lookups != 0 else None)


def _final_roles(pipeline, queries):
    rows = _complete_rows(pipeline, "cases", queries)
    counts = {"small": 0, "large": 0}
    if rows is not None:
        for row in rows:
            index, attempts = row.get("final_attempt_index"), row.get("attempts")
            if (type(index) is not int or not isinstance(attempts, list)
                    or not 0 <= index < len(attempts) or not isinstance(attempts[index], dict)):
                break
            role = attempts[index].get("role")
            if role not in ("small", "large"):
                break
            counts[role] += 1
        else:
            return counts
    return {"small": None, "large": None}


def benchmark_metrics(report):
    """Keep pair outcomes, repeated lookups, queries and model attempts separate."""
    pipeline = report["pipeline_benchmark"]
    routing = pipeline["routing_metrics"]
    queries = routing["query_count"]
    escalation = _group(pipeline, "escalation_metrics")
    cases = _group(routing, "classifications")
    simple, complex_count = (_count(cases, k, queries) for k in ("simple", "complex"))
    _partition((simple, complex_count), queries)
    initial = _group(routing, "initial_selections")
    small, large = (_count(initial, k, queries) for k in ("small", "large"))
    _partition((small, large), queries)
    attempts = _group(routing, "all_attempt_roles")
    attempt_counts = {k: _count(attempts, k) for k in ("small", "large")}
    total_attempts = _count(escalation, "total_model_attempts")
    _partition(attempt_counts.values(), total_attempts)
    escalations = _count(escalation, "escalations_triggered", queries)
    if escalations is not None and total_attempts is not None and total_attempts != queries + escalations:
        raise ValueError("Inconsistent escalation attempt total.")
    successes = _count(escalation, "successful_escalations", queries if escalations is None else escalations)
    accepted = _count(escalation, "initial_answers_accepted", queries)
    if accepted is not None and escalations is not None and accepted + escalations > queries:
        raise ValueError("Accepted initial answers cannot also be escalated.")
    return dict(cache=sorted((_cache(c) for c in report["cache_benchmarks"]), key=lambda c: c["t"]),
                comparison=project_comparison(report.get("cost_comparison")),
                queries=queries, simple=simple, complex=complex_count, small=small, large=large,
                attempts=attempt_counts, final=_final_roles(pipeline, queries),
                escalations=escalations, successes=successes, accepted=accepted,
                escalationRate=_rate(escalations, queries), acceptanceRate=_rate(accepted, queries),
                successRate=_rate(successes, escalations),
                passed=_count(_group(pipeline, "quality_metrics"), "final_passes", queries))


def historical_metrics(records, *, start=None, end=None, model=None, currency="USD", rate="1"):
    """Aggregate recorded costs with Decimal; optional filters use UTC calendar days."""
    if currency not in ("USD", "EUR", "INR", "GBP") or amount(rate) <= 0:
        raise ValueError("Invalid currency configuration.")
    if start:
        date.fromisoformat(start)
    if end:
        date.fromisoformat(end)
    if start and end and start > end:
        raise ValueError("Invalid date range.")
    selected = [r for r in records
                if (not start or r["timestamp_utc"][:10] >= start)
                and (not end or r["timestamp_utc"][:10] <= end)
                and (not model or model == "All models" or r["model"] == model)]
    count = len(selected)

    def summarize(rows):
        result = summarize_requests(rows, query_count=count)
        costs = {**result["cost"], "display": cost_display(result["cost"], currency, rate),
                 "requests": result["requests"], "components": result["components"],
                 "generation_calls": sum(row["request_kind"] != "embedding" for row in rows)}
        for component in costs["components"].values():
            component["display"] = cost_display(component, currency, rate)
        return costs

    summary = summarize(selected)
    summary["by_model"] = [
        {"model": name, **summarize([r for r in selected if r["model"] == name])}
        for name in sorted({r["model"] for r in selected})]
    summary["embedding"] = summarize([r for r in selected if r["request_kind"] == "embedding"])
    summary["daily"] = []
    for day in sorted({r["timestamp_utc"][:10] for r in selected}):
        daily = summarize([r for r in selected if r["timestamp_utc"][:10] == day])
        cost = daily["estimated_cost_usd"]
        summary["daily"].append({"date": day, "cost": None if cost is None else str(multiply(cost, rate)),
                                 "display": daily["display"]["estimated_cost_usd"]})
    return summary
