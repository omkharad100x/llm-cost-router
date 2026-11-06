"""Measure synchronous Gemini calls without retaining input or response text."""

from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

from money import COST_FIELDS, add, amount, multiply


def load_pricing(path, *, as_of=None):
    """Load standard rates; entries past their documented end date are unavailable."""
    with Path(path).open(encoding="utf-8") as file:
        pricing = json.load(file, parse_float=amount)
    if not isinstance(pricing, dict):
        raise ValueError("Pricing must be a model-to-rates object.")
    today = as_of if as_of is not None else datetime.now(timezone.utc).date()
    if type(today) is not date:
        raise ValueError("Pricing date must be a calendar date.")
    result = {}
    for model, rates in pricing.items():
        if not isinstance(rates, dict) or not {
            "input_usd_per_million", "output_usd_per_million"
        }.issubset(rates):
            raise ValueError("Each model needs input and output pricing.")
        rates = rates.copy()
        for key in ("input_usd_per_million", "output_usd_per_million", "cached_input_usd_per_million"):
            if key in rates:
                value = amount(rates[key])
                if value < 0:
                    raise ValueError("Pricing rates must be nonnegative.")
                rates[key] = str(value)
        expires = rates.get("valid_through")
        if expires is not None and today > date.fromisoformat(expires):
            continue
        result[model] = rates
    return result


def token_count(usage, field):
    """Keep missing or invalid counts distinct from a reported zero."""
    value = getattr(usage, field, None)
    if type(value) is int and value >= 0:
        return value
    return None


def _token_charge(tokens, rates, field):
    if tokens is None or rates is None or rates.get(field) is None:
        return None
    price = amount(rates[field])
    if price < 0:
        raise ValueError("Pricing rates must be nonnegative.")
    return str(multiply(multiply(tokens, price), "0.000001"))


def usage_metrics(response, rates):
    """Price standard text usage while preserving unknown counters and components."""
    usage = getattr(response, "usage_metadata", None)
    sdk_fields = {
        "input_tokens": "prompt_token_count", "output_tokens": "candidates_token_count",
        "thinking_tokens": "thoughts_token_count", "total_tokens": "total_token_count",
        "cached_input_tokens": "cached_content_token_count", "tool_use_tokens": "tool_use_prompt_token_count",
    }
    result = {name: token_count(usage, field) for name, field in sdk_fields.items()}
    prompt, output, thoughts, total, cached, tools = (result[k] for k in sdk_fields)
    malformed = any(getattr(usage, field, None) is not None and result[name] is None
                    for name, field in sdk_fields.items())
    billed = None
    inconsistent = malformed
    if output is not None and not tools and not malformed:
        if thoughts is not None:
            billed = output + thoughts
            inconsistent = prompt is not None and total is not None and total != prompt + billed
        elif prompt is not None and total is not None and total >= prompt + output:
            billed = total - prompt
        elif prompt is not None and total is not None:
            inconsistent = True
    if cached is not None and prompt is not None and cached > prompt:
        inconsistent = True
    if inconsistent:
        billed = None
    result.update(billed_output_tokens=billed, estimated_cost_usd=None,
                  cost_status="usage_unavailable", pricing_version=rates.get("pricing_version") if rates else None,
                  **dict.fromkeys(COST_FIELDS))
    if rates is None:
        result["cost_status"] = "pricing_unavailable"
    elif inconsistent:
        result["cost_status"] = "usage_inconsistent"
    elif tools:
        # Tool and grounding billing is outside this text-only application's policy.
        result["cost_status"] = "usage_unavailable"
    else:
        # A zero-size prompt cannot contain cached tokens. All other missing cache counts stay unknown.
        cache_count = 0 if prompt == 0 else cached
        ordinary = prompt - cache_count if prompt is not None and cache_count is not None else None
        result["input_cost_usd"] = _token_charge(ordinary, rates, "input_usd_per_million")
        result["cached_input_cost_usd"] = ("0" if cache_count == 0 else
                                          _token_charge(cache_count, rates, "cached_input_usd_per_million"))
        result["output_cost_usd"] = _token_charge(billed, rates, "output_usd_per_million")
        if cache_count and rates.get("cached_input_usd_per_million") is None:
            result["cost_status"] = "cached_pricing_unavailable"
        elif any(rates.get(key) is None for key in ("input_usd_per_million", "output_usd_per_million")):
            result["cost_status"] = "pricing_unavailable"
        elif all(result[k] is not None for k in COST_FIELDS):
            result["estimated_cost_usd"] = str(add(result[k] for k in COST_FIELDS))
            result["cost_status"] = "estimated"
    return result


def embedding_usage_metrics(response, rates):
    """Use embedding token statistics only; character billing stays unpriced."""
    embeddings = getattr(response, "embeddings", None)
    tokens = None
    if isinstance(embeddings, (list, tuple)) and len(embeddings) == 1:
        statistics = getattr(embeddings[0], "statistics", None)
        count = getattr(statistics, "token_count", None)
        if type(count) is int and count >= 0:
            tokens = count
        elif type(count) is float and math.isfinite(count) and 0 <= count <= 2**53 and count.is_integer():
            tokens = int(count)
    cost = _token_charge(tokens, rates, "input_usd_per_million")
    status = "estimated" if cost is not None else "usage_unavailable"
    if rates is None or rates.get("input_usd_per_million") is None:
        status = "pricing_unavailable"
    return {
        "input_tokens": tokens, "output_tokens": None,
        "estimated_cost_usd": cost, "cost_status": status,
        "input_cost_usd": cost, "cached_input_cost_usd": "0", "output_cost_usd": "0",
        "usage_source": "embedding_statistics" if tokens is not None else None,
        "pricing_version": rates.get("pricing_version") if rates else None,
    }


class JsonlRecorder:
    """Append metrics only, one JSON record per SDK invocation."""

    def __init__(self, path):
        self.path = Path(path)

    def __call__(self, record):
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")


class InstrumentedGenerator:
    """Return original responses and re-raise original provider exceptions."""

    def __init__(self, client, pricing, recorder):
        self.client = client
        self.pricing = pricing
        self.recorder = recorder

    def generate_content(self, *, model, request_kind, **kwargs):
        return self._request(
            self.client.models.generate_content, usage_metrics,
            model, request_kind, kwargs,
        )

    def generate_content_with_metrics(self, *, model, request_kind, **kwargs):
        """Return (SDK response, metrics snapshot) from one instrumented call.

        The snapshot can be None if metric extraction fails. Recorder failure
        does not discard an already extracted snapshot. No shared last-call state.
        """
        return self._request(
            self.client.models.generate_content, usage_metrics,
            model, request_kind, kwargs, include_metrics=True,
        )

    def embed_content(self, *, model, **kwargs):
        return self._request(
            self.client.models.embed_content, embedding_usage_metrics,
            model, "embedding", kwargs,
        )

    def _request(self, method, extract_usage, model, request_kind, kwargs, *, include_metrics=False):
        timestamp = datetime.now(timezone.utc).isoformat()
        response = None
        status = "error"
        snapshot = None
        started = time.perf_counter()
        try:
            response = method(model=model, **kwargs)
            status = "success"
        finally:
            latency_ms = (time.perf_counter() - started) * 1000
            try:
                record = {
                    "timestamp_utc": timestamp,
                    "model": model,
                    "request_kind": request_kind,
                    "status": status,
                    "latency_ms": latency_ms,
                    **extract_usage(response, self.pricing.get(model)),
                }
                if include_metrics:
                    snapshot = record.copy()
                self.recorder(record)
            except Exception:
                # Never let metrics failures alter responses, retries, or memory.
                print("Warning: request metrics could not be recorded.", file=sys.stderr)

        return (response, snapshot) if include_metrics else response
