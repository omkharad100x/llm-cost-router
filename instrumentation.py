"""Measure synchronous Gemini calls without retaining input or response text."""

from datetime import datetime, timezone
from decimal import Decimal
import json
import math
from pathlib import Path
import sys
import time


def load_pricing(path):
    """Load configurable USD rates per million tokens; unknown models stay unpriced."""
    with Path(path).open(encoding="utf-8") as file:
        pricing = json.load(file)
    if not isinstance(pricing, dict):
        raise ValueError("Pricing must be a model-to-rates object.")
    for rates in pricing.values():
        if not isinstance(rates, dict) or not {
            "input_usd_per_million", "output_usd_per_million"
        }.issubset(rates):
            raise ValueError("Each model needs input and output pricing.")
        for name in (
            "input_usd_per_million",
            "output_usd_per_million",
            "cached_input_usd_per_million",
        ):
            if name in rates:
                rate = Decimal(str(rates[name]))
                if not rate.is_finite() or rate < 0:
                    raise ValueError("Pricing rates must be finite and nonnegative.")
    return pricing


def token_count(usage, field):
    """Keep missing or invalid counts distinct from a reported zero."""
    value = getattr(usage, field, None)
    if type(value) is int and value >= 0:
        return value
    return None


def usage_metrics(response, rates):
    """Extract SDK fields and estimate standard text generation cost."""
    usage = getattr(response, "usage_metadata", None)
    input_tokens = token_count(usage, "prompt_token_count")
    output_tokens = token_count(usage, "candidates_token_count")
    thinking_tokens = token_count(usage, "thoughts_token_count")
    total_tokens = token_count(usage, "total_token_count")
    cached_tokens = token_count(usage, "cached_content_token_count")
    tool_tokens = token_count(usage, "tool_use_prompt_token_count")
    billed_output = None

    # This CLI has no tools. Do not price unexpected tool usage as ordinary text.
    if input_tokens is not None and output_tokens is not None and not tool_tokens:
        if thinking_tokens is not None:
            billed_output = output_tokens + thinking_tokens
            if total_tokens is not None and total_tokens != input_tokens + billed_output:
                billed_output = None
        elif total_tokens is not None and total_tokens >= input_tokens + output_tokens:
            # Infer billable output from the SDK total, while preserving raw nulls.
            billed_output = total_tokens - input_tokens

    cost = None
    cost_status = "usage_unavailable"
    if rates is None:
        cost_status = "pricing_unavailable"
    elif billed_output is not None:
        cached = cached_tokens if cached_tokens is not None else 0
        if cached > input_tokens:
            cost_status = "usage_inconsistent"
        elif cached and "cached_input_usd_per_million" not in rates:
            cost_status = "cached_pricing_unavailable"
        else:
            normal_rate = Decimal(str(rates["input_usd_per_million"]))
            output_rate = Decimal(str(rates["output_usd_per_million"]))
            cache_rate = Decimal(str(rates.get("cached_input_usd_per_million", 0)))
            amount = (
                (input_tokens - cached) * normal_rate
                + cached * cache_rate
                + billed_output * output_rate
            ) / Decimal(1_000_000)
            cost = str(amount)
            cost_status = "estimated"

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
        "billed_output_tokens": billed_output,
        "total_tokens": total_tokens,
        "cached_input_tokens": cached_tokens,
        "estimated_cost_usd": cost,
        "cost_status": cost_status,
        "pricing_version": rates.get("pricing_version") if rates else None,
    }


def embedding_usage_metrics(response, rates):
    """Use SDK embedding statistics only; never guess tokens from characters."""
    embeddings = getattr(response, "embeddings", None)
    input_tokens = None
    if isinstance(embeddings, (list, tuple)) and len(embeddings) == 1:
        statistics = getattr(embeddings[0], "statistics", None)
        count = getattr(statistics, "token_count", None)
        # The SDK types this field as float, even for integral token counts.
        if type(count) in (int, float) and math.isfinite(count) and count >= 0:
            if count == int(count):
                input_tokens = int(count)
    cost = None
    cost_status = "usage_unavailable"
    if rates is None:
        cost_status = "pricing_unavailable"
    elif input_tokens is not None:
        cost = str(
            Decimal(input_tokens) * Decimal(str(rates["input_usd_per_million"]))
            / Decimal(1_000_000)
        )
        cost_status = "estimated"
    return {
        "input_tokens": input_tokens,
        "output_tokens": None,
        "estimated_cost_usd": cost,
        "cost_status": cost_status,
        "usage_source": "embedding_statistics" if input_tokens is not None else None,
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
