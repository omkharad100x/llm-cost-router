"""Complexity-first routing with explicit pricing availability and shared model calls."""

from collections.abc import Mapping
from dataclasses import dataclass

from complexity_config import ComplexityConfig, DEFAULT_COMPLEXITY_CONFIG
from gemini_models import GeminiModels, ModelResponse
from model_config import ModelConfig
from query_complexity import ComplexityResult, classify_query


@dataclass(frozen=True)
class RoutingConfig:
    simple_role: str = "small"
    complex_role: str = "large"
    complexity: ComplexityConfig = DEFAULT_COMPLEXITY_CONFIG

    def __post_init__(self):
        for role in (self.simple_role, self.complex_role):
            if not isinstance(role, str) or role not in ("small", "large"):
                raise ValueError("Routing roles must be small or large.")
        if not isinstance(self.complexity, ComplexityConfig):
            raise ValueError("complexity must be a ComplexityConfig instance.")


DEFAULT_ROUTING_CONFIG = RoutingConfig()


@dataclass(frozen=True)
class RoutingDecision:
    selected_role: str
    reason: str
    pricing_status: str
    policy: str = "complexity-v1"
    cost_optimized: bool = False


@dataclass(frozen=True)
class RoutedGeneration:
    classification: ComplexityResult
    decision: RoutingDecision
    response: ModelResponse


class RoutingError(RuntimeError):
    """A safe routing failure, optionally retaining the attempted decision."""

    def __init__(self, message, *, decision=None):
        super().__init__(message)
        self.decision = decision


class ClassificationError(RoutingError):
    """Classification failed before any generation was attempted."""


class RoutingGenerationError(RoutingError):
    """The selected generation failed; no other role was attempted."""


class ModelRouter:
    """Reuse an existing GeminiModels instance; do not create clients or load secrets."""

    def __init__(self, models, *, config=DEFAULT_ROUTING_CONFIG):
        if not isinstance(models, GeminiModels):
            raise ValueError("models must be a GeminiModels instance.")
        if not isinstance(config, RoutingConfig):
            raise ValueError("config must be a RoutingConfig instance.")
        if not isinstance(models.config, ModelConfig):
            raise ValueError("The model layer requires a ModelConfig instance.")
        if not isinstance(models.generator.pricing, Mapping):
            raise ValueError("The model layer requires a pricing mapping.")
        self.models = models
        self.config = config

    def generate(self, query, *, generation_config=None):
        """Classify once, select once, invoke once, and preserve the model response."""
        try:
            classification = classify_query(query, config=self.config.complexity)
            if (
                not isinstance(classification, ComplexityResult)
                or classification.label not in ("simple", "complex")
                or type(classification.score) is not int or classification.score < 0
            ):
                raise ValueError
        except Exception:
            raise ClassificationError("Query classification failed.") from None

        role = (
            self.config.simple_role if classification.label == "simple"
            else self.config.complex_role
        )
        model = self.models.config.small_model if role == "small" else self.models.config.large_model
        # This records rate availability only; it is not a price comparison.
        configured = self.models.generator.pricing.get(model) is not None
        pricing_status = "configured" if configured else "unavailable"
        pricing_reason = (
            "Pricing is configured; estimated cost depends on returned usage."
            if configured else
            "Pricing is unavailable; estimated cost remains unknown."
        )
        decision = RoutingDecision(
            selected_role=role,
            reason=(
                f"Complexity label {classification.label} (score {classification.score}) "
                f"maps to the {role} role under the configured policy. "
                + pricing_reason + " Cost optimality has not been evaluated."
            ),
            pricing_status=pricing_status,
        )
        try:
            response = self.models.generate(
                query, role=role, generation_config=generation_config,
            )
            if (
                not isinstance(response, ModelResponse)
                or response.role != role or response.model != model
            ):
                raise ValueError
        except Exception:
            # Suppress provider or injected component messages and exception chains.
            raise RoutingGenerationError(
                "Selected model generation failed.", decision=decision,
            ) from None
        return RoutedGeneration(classification, decision, response)
