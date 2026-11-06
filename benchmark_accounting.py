"""Shared Decimal accounting for recorded benchmark requests."""

from collections import Counter
from decimal import Decimal
import math


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _request_totals(records, query_count):
    def total(field):
        values = [record.get(field) for record in records]
        return sum(values) if all(value is not None for value in values) else None

    known_costs = [Decimal(record["estimated_cost_usd"]) for record in records
                   if record.get("estimated_cost_usd") is not None]
    unknown = len(records) - len(known_costs)
    subtotal = sum(known_costs, Decimal(0))
    total_cost = None if unknown else subtotal
    statuses = Counter(record.get("cost_status", "metrics_unavailable") for record in records)
    status = "estimated" if records else "no_requests"
    if unknown:
        status = ("pricing_unavailable" if any(
            name in statuses for name in ("pricing_unavailable", "cached_pricing_unavailable")
        ) else "incomplete")
    latency = total("latency_ms")
    return {
        "requests": len(records),
        "cost": {
            "estimated_cost_usd": str(total_cost) if total_cost is not None else None,
            "average_cost_per_query_usd": str(total_cost / query_count)
            if total_cost is not None and query_count else None,
            "known_cost_subtotal_usd": str(subtotal), "unknown_cost_requests": unknown,
            "complete": unknown == 0, "status": status, "attempt_statuses": dict(sorted(statuses.items())),
        },
        "usage": {field: total(field) for field in
                  ("input_tokens", "output_tokens", "thinking_tokens", "billed_output_tokens")},
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
        for field in ("input_tokens", "output_tokens", "thinking_tokens", "billed_output_tokens"):
            value = record.get(field)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("Token metrics must be nonnegative integers or None.")
        value = record.get("latency_ms")
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            raise ValueError("Latency must be finite and nonnegative or None.")
        value = record.get("estimated_cost_usd")
        if value is not None:
            try:
                amount = Decimal(value)
                if not amount.is_finite() or amount < 0:
                    raise ValueError
            except (ValueError, TypeError, ArithmeticError):
                raise ValueError("Cost must be a finite nonnegative decimal or None.") from None
    return {
        **_request_totals(records, query_count),
        "by_model": {model: _request_totals([r for r in records if r["model"] == model], query_count)
                     for model in sorted({record["model"] for record in records})},
    }


