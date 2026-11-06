"""Shared Decimal accounting for recorded benchmark requests."""

from collections import Counter
import math

from money import COST_FIELDS, add, amount, divide


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _request_totals(records, query_count):
    def total(field):
        values = [record.get(field) for record in records]
        return sum(values) if all(value is not None for value in values) else None

    known_costs = [amount(record["estimated_cost_usd"]) for record in records
                   if record.get("estimated_cost_usd") is not None]
    unknown = len(records) - len(known_costs)
    subtotal = add(known_costs)
    total_cost = None if unknown else subtotal
    statuses = Counter(record.get("cost_status", "metrics_unavailable") for record in records)
    status = "estimated" if records else "no_requests"
    if unknown:
        status = ("pricing_unavailable" if any(
            name in statuses for name in ("pricing_unavailable", "cached_pricing_unavailable")
        ) else "incomplete")
    components = {}
    for field in COST_FIELDS:
        values = [record.get(field) for record in records]
        known = add(v for v in values if v is not None)
        components[field] = {
            "estimated_cost_usd": str(known) if all(v is not None for v in values) else None,
            "known_cost_subtotal_usd": str(known),
            "unknown_cost_requests": sum(v is None for v in values),
        }
    latency = total("latency_ms")
    return {
        "requests": len(records),
        "cost": {
            "estimated_cost_usd": str(total_cost) if total_cost is not None else None,
            "average_cost_per_query_usd": str(divide(total_cost, query_count))
            if total_cost is not None and query_count else None,
            "known_cost_subtotal_usd": str(subtotal), "unknown_cost_requests": unknown,
            "complete": unknown == 0, "status": status, "attempt_statuses": dict(sorted(statuses.items())),
        },
        "components": components,
        "usage": {field: total(field) for field in
                  ("input_tokens", "output_tokens", "thinking_tokens", "billed_output_tokens",
                   "cached_input_tokens", "tool_use_tokens", "total_tokens")},
        "latency": {
            "total_ms": latency,
            "average_per_request_ms": ratio(latency, len(records)) if latency is not None else None,
            "average_per_query_ms": ratio(latency, query_count) if latency is not None else None,
            "source": "measured local mock SDK call overhead; not provider latency",
        },
    }


def summarize_requests(records, *, query_count):
    """Aggregate instrumentation records without inventing tokens or repricing."""
    if type(query_count) is not int or query_count < 0:
        raise ValueError("query_count must be a nonnegative integer.")
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("model"), str):
            raise ValueError("Expected instrumentation records with model names.")
        for field in ("input_tokens", "output_tokens", "thinking_tokens", "billed_output_tokens",
                   "cached_input_tokens", "tool_use_tokens", "total_tokens"):
            value = record.get(field)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("Token metrics must be nonnegative integers or None.")
        prompt, cached = record.get("input_tokens"), record.get("cached_input_tokens")
        output, thoughts, billed = (record.get(k) for k in ("output_tokens", "thinking_tokens", "billed_output_tokens"))
        if record.get("estimated_cost_usd") is not None and prompt is not None and cached is not None and cached > prompt:
            raise ValueError("Cached tokens exceed the prompt.")
        if record.get("estimated_cost_usd") is not None and billed is not None and output is not None and (
                billed < output or (thoughts is not None and billed != output + thoughts)):
            raise ValueError("Inconsistent billed output.")
        if record.get("tool_use_tokens") and record.get("estimated_cost_usd") is not None:
            raise ValueError("Tool usage has no supported text-only price.")
        value = record.get("latency_ms")
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            raise ValueError("Latency must be finite and nonnegative or None.")
        for field in ("estimated_cost_usd", *COST_FIELDS):
            value = record.get(field)
            if value is not None and amount(value) < 0:
                raise ValueError("Costs must be nonnegative.")
        cost = record.get("estimated_cost_usd")
        if (record.get("cost_status") == "estimated") != (cost is not None):
            raise ValueError("Cost status disagrees with cost availability.")
        parts = [record.get(field) for field in COST_FIELDS]
        if cost is not None and all(v is not None for v in parts) and add(parts) != amount(cost):
            raise ValueError("Cost components disagree with the recorded total.")
    return {
        **_request_totals(records, query_count),
        "by_model": {model: _request_totals([r for r in records if r["model"] == model], query_count)
                     for model in sorted({record["model"] for record in records})},
    }


