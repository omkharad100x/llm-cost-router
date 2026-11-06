"""Validate offline comparison ledgers and project Decimal costs for both UIs."""

from decimal import Decimal
import math
import re

from benchmark_accounting import summarize_requests


UNKNOWN_COST = {"pricing_unavailable", "usage_unavailable", "usage_inconsistent",
                "cached_pricing_unavailable", "metrics_unavailable"}
TOKEN_FIELDS = ("input_tokens", "output_tokens", "thinking_tokens", "billed_output_tokens")


def _record(record, models):
    if not isinstance(record, dict) or record.get("status") != "success":
        raise ValueError("Expected a successful recorded fixture call.")
    phase = record.get("phase")
    role = {"small_model": "small", "large_model": "large", "embedding": "embedding"}.get(record.get("request_kind"))
    if role is None or record.get("model") != models[role]:
        raise ValueError("Recorded model and role disagree.")
    if phase not in ("baseline", "initial", "escalation", "embedding"):
        raise ValueError("Invalid comparison request phase.")
    if (phase == "embedding") != (role == "embedding") or (
            phase in ("baseline", "escalation") and role != "large"):
        raise ValueError("Invalid comparison model phase.")
    for field in TOKEN_FIELDS:
        value = record.get(field)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("Invalid token count.")
    cost, status = record.get("estimated_cost_usd"), record.get("cost_status")
    if cost is None:
        if status not in UNKNOWN_COST:
            raise ValueError("Missing cost status.")
    else:
        if not isinstance(cost, str) or len(cost) > 64:
            raise ValueError("Costs must be decimal strings.")
        amount = Decimal(cost)
        if not amount.is_finite() or amount < 0 or abs(amount.adjusted()) > 100 or status != "estimated":
            raise ValueError("Invalid cost estimate.")
        if record.get("input_tokens") is None or (role != "embedding" and record.get("billed_output_tokens") is None):
            raise ValueError("Estimated costs require measured usage.")
    # Whitelist fields; arbitrary report text never reaches the dashboard.
    return dict(model=record["model"], phase=phase, estimated_cost_usd=cost, cost_status=status,
                **{k: record.get(k) for k in TOKEN_FIELDS})


def _summary(records, count):
    summary = summarize_requests(records, query_count=count)
    return {**summary["cost"], "requests": len(records),
            "generation_calls": sum(r["phase"] != "embedding" for r in records)}


def comparison_metrics(payload):
    """A versioned ledger pairs every query's baseline and optimized attempts."""
    if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("Unsupported comparison schema.")
    if payload.get("methodology") != "paired-routing-fixtures-v1":
        raise ValueError("Unsupported comparison methodology.")
    models = payload["models"]
    if (not isinstance(models, dict) or set(models) != {"small", "large", "embedding"}
            or len(set(models.values())) != 3
            or any(not isinstance(m, str) or not re.fullmatch(r"gemini-[A-Za-z0-9._-]{1,120}", m) for m in models.values())):
        raise ValueError("Invalid comparison models.")
    threshold = payload["threshold"]
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError("Invalid comparison threshold.")
    queries = payload["queries"]
    if not isinstance(queries, list):
        raise ValueError("Missing paired workload.")
    ids = set()
    baseline, optimized = [], []
    outcomes = dict(safe_hit=0, false_hit=0, unreviewed_hit=0, miss=0)
    for query in queries:
        if not isinstance(query, dict) or not isinstance(query.get("id"), str) or not query["id"] or query["id"] in ids:
            raise ValueError("Invalid or duplicate paired query.")
        ids.add(query["id"])
        outcome = query["cache_outcome"]
        if outcome not in outcomes:
            raise ValueError("Missing reviewed cache outcome.")
        outcomes[outcome] += 1
        base, opt = query["baseline"], query["optimized"]
        if not isinstance(base, list) or len(base) != 1 or not isinstance(opt, list):
            raise ValueError("Both scenarios must cover every query.")
        base = [_record(r, models) for r in base]
        opt = [_record(r, models) for r in opt]
        if base[0]["phase"] != "baseline" or any(r["phase"] == "baseline" for r in opt):
            raise ValueError("Invalid scenario assignment.")
        generation = [r["phase"] for r in opt if r["phase"] != "embedding"]
        if outcome == "miss":
            if generation not in (["initial"], ["initial", "escalation"]):
                raise ValueError("Cache misses require recorded generation.")
        elif generation or not opt:
            raise ValueError("Cache hits require lookup evidence and no generation.")
        baseline.extend(base)
        optimized.extend(opt)
    count = len(queries)
    base, opt = _summary(baseline, count), _summary(optimized, count)
    comparable = base["complete"] and opt["complete"] and not outcomes["false_hit"] and not outcomes["unreviewed_hit"]
    savings = percentage = None
    if comparable:
        baseline_cost = Decimal(base["estimated_cost_usd"])
        savings = baseline_cost - Decimal(opt["estimated_cost_usd"])
        if baseline_cost:
            percentage = savings * 100 / baseline_cost
    issues = []
    for model in sorted(set(r["model"] for r in baseline + optimized)):
        records = [r for r in baseline + optimized if r["model"] == model]
        if any(r["cost_status"] in ("pricing_unavailable", "cached_pricing_unavailable") for r in records):
            issues.append("Pricing unavailable for " + model + ".")
        if any(r["input_tokens"] is None or (r["phase"] != "embedding" and r["billed_output_tokens"] is None) for r in records):
            issues.append("Token usage unavailable for " + model + ".")
        if any(r["cost_status"] in ("metrics_unavailable", "usage_inconsistent") for r in records):
            issues.append("Incomplete request accounting for " + model + ".")
    if outcomes["false_hit"] or outcomes["unreviewed_hit"]:
        issues.append("Savings withheld because the workload contains unsafe or unreviewed cache reuse.")
    return dict(status="ready", query_count=count, threshold=threshold, baseline=base, optimized=opt,
                savings_usd=str(savings) if savings is not None else None,
                savings_percentage=str(percentage) if percentage is not None else None,
                safe_calls_avoided=outcomes["safe_hit"], cache_outcomes=outcomes, issues=issues,
                escalation=_summary([r for r in optimized if r["phase"] == "escalation"], count),
                embedding=_summary([r for r in optimized if r["phase"] == "embedding"], count),
                by_model=[dict(model=model, baseline=_summary([r for r in baseline if r["model"] == model], count),
                               optimized=_summary([r for r in optimized if r["model"] == model], count))
                          for model in sorted(set(models.values()))])


def project_comparison(payload):
    """Older or corrupt comparison ledgers leave the other dashboard views usable."""
    if payload is None:
        return {"status": "missing", "issues": ["This report has no paired cost workload. Run the offline benchmarks to create one."]}
    try:
        return comparison_metrics(payload)
    except (ValueError, KeyError, TypeError, ArithmeticError, RecursionError):
        return {"status": "invalid", "issues": ["The paired cost workload could not be validated. Regenerate the offline report."]}
