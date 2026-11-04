"""Explicit small/large Gemini calls through an existing InstrumentedGenerator."""

from dataclasses import dataclass

from instrumentation import InstrumentedGenerator
from model_config import ModelConfig, load_model_config


class ModelCallError(RuntimeError):
    """A provider call failed; its raw error is deliberately not displayed."""


class ModelResponseError(ModelCallError):
    """The provider returned no usable text response."""


@dataclass(frozen=True)
class ModelResponse:
    text: str
    model: str
    role: str
    input_tokens: int | None
    output_tokens: int | None
    thinking_tokens: int | None
    billed_output_tokens: int | None
    latency_ms: float | None
    estimated_cost_usd: str | None
    cost_status: str
    timestamp_utc: str | None


def _response_text(response):
    """Read first-candidate text parts without SDK text-property diagnostic logs."""
    try:
        candidates = getattr(response, "candidates", None)
        if not isinstance(candidates, (list, tuple)) or not candidates:
            raise ValueError
        parts = getattr(getattr(candidates[0], "content", None), "parts", None)
        if not isinstance(parts, (list, tuple)):
            raise ValueError
        texts = []
        for part in parts:
            if getattr(part, "thought", False):
                continue
            text = getattr(part, "text", None)
            if text is not None:
                if not isinstance(text, str):
                    raise ValueError
                texts.append(text)
        text = "".join(texts).strip()
        if not text:
            raise ValueError
        return text
    except Exception:
        raise ModelResponseError("Gemini returned no usable text response.") from None


class GeminiModels:
    """Inject the shared instrumented client; callers explicitly choose a role."""

    def __init__(self, generator, *, config=None):
        if not isinstance(generator, InstrumentedGenerator):
            raise ValueError("generator must be an InstrumentedGenerator instance.")
        self.generator = generator
        self.config = load_model_config() if config is None else config
        if not isinstance(self.config, ModelConfig):
            raise ValueError("config must be a ModelConfig instance.")

    def generate(self, query, *, role, generation_config=None):
        """One synchronous text request; no classification, fallback, or added retry."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a nonempty text string.")
        if not isinstance(role, str) or role not in ("small", "large"):
            raise ValueError("role must be small or large.")
        model = self.config.small_model if role == "small" else self.config.large_model
        kwargs = {"contents": query}
        if generation_config is not None:
            kwargs["config"] = generation_config
        try:
            response, metrics = self.generator.generate_content_with_metrics(
                model=model, request_kind=f"{role}_model", **kwargs,
            )
        except Exception:
            # Suppress raw SDK messages and traceback chains that can contain
            # request URLs, credentials, input text, or provider diagnostics.
            raise ModelCallError("Gemini generation request failed.") from None
        text = _response_text(response)
        metrics = metrics or {}
        return ModelResponse(
            text=text, model=model, role=role,
            input_tokens=metrics.get("input_tokens"),
            output_tokens=metrics.get("output_tokens"),
            thinking_tokens=metrics.get("thinking_tokens"),
            billed_output_tokens=metrics.get("billed_output_tokens"),
            latency_ms=metrics.get("latency_ms"),
            estimated_cost_usd=metrics.get("estimated_cost_usd"),
            cost_status=metrics.get("cost_status", "metrics_unavailable"),
            timestamp_utc=metrics.get("timestamp_utc"),
        )
