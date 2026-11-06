"""One quality-based escalation using the existing router, models, and evaluator."""

from dataclasses import dataclass, replace
from money import add
import math
from numbers import Real

from answer_evaluator import EvaluationResult, evaluate_answer
from evaluation_config import EvaluationConfig, DEFAULT_EVALUATION_CONFIG
from gemini_models import ModelResponse
from model_router import (
    ClassificationError, ModelRouter, RoutedGeneration,
    RoutingGenerationError,
)


@dataclass(frozen=True)
class GenerationAttempt:
    """A model invocation, including incomplete attempts retained on errors."""

    role: str
    model: str
    response: ModelResponse | None
    evaluation: EvaluationResult | None


@dataclass(frozen=True)
class AggregateMetrics:
    """Complete totals only; each unavailable total is None, never a partial sum."""

    input_tokens: int | None
    output_tokens: int | None
    thinking_tokens: int | None
    billed_output_tokens: int | None
    latency_ms: float | None
    estimated_cost_usd: str | None
    cost_status: str
    attempt_cost_statuses: tuple[str, ...]


def _aggregate(attempts):
    responses = tuple(attempt.response for attempt in attempts)

    def total(field):
        values = [getattr(response, field) if response is not None else None
                  for response in responses]
        return sum(values) if values and all(value is not None for value in values) else None

    statuses = tuple(
        response.cost_status if response is not None else "response_unavailable"
        for response in responses
    )
    costs = [response.estimated_cost_usd if response is not None else None
             for response in responses]
    cost = None
    if not responses:
        status = "not_requested"
    elif all(value is not None for value in costs):
        # Sum the instrumented USD amounts, never recalculate rates or round per call.
        cost = str(add(costs))
        status = "estimated"
    elif any(value in ("pricing_unavailable", "cached_pricing_unavailable") for value in statuses):
        status = "pricing_unavailable"
    else:
        status = "incomplete"
    return AggregateMetrics(
        input_tokens=total("input_tokens"), output_tokens=total("output_tokens"),
        thinking_tokens=total("thinking_tokens"),
        billed_output_tokens=total("billed_output_tokens"),
        latency_ms=total("latency_ms"), estimated_cost_usd=cost,
        cost_status=status, attempt_cost_statuses=statuses,
    )


@dataclass(frozen=True)
class EscalationResult:
    """Original component results are retained without duplicating their fields."""

    query: str
    initial_routing: RoutedGeneration
    attempts: tuple[GenerationAttempt, ...]
    escalated: bool
    escalation_reason: str
    escalation_role: str | None
    final_attempt_index: int
    totals: AggregateMetrics

    @property
    def total_attempts(self):
        return len(self.attempts)

    @property
    def initial_response(self):
        return self.attempts[0].response

    @property
    def initial_evaluation(self):
        return self.attempts[0].evaluation

    @property
    def escalation_response(self):
        return self.attempts[1].response if self.escalated else None

    @property
    def escalation_evaluation(self):
        return self.attempts[1].evaluation if self.escalated else None

    @property
    def final_response(self):
        return self.attempts[self.final_attempt_index].response

    @property
    def final_evaluation(self):
        return self.attempts[self.final_attempt_index].evaluation


class EscalationError(RuntimeError):
    """Sanitized failure with partial results; no fallback answer is silently chosen."""

    def __init__(self, *, stage, attempts=(), initial_routing=None, routing_decision=None):
        super().__init__("Escalation pipeline failed during " + stage.replace("_", " ") + ".")
        self.stage = stage
        self.attempts = tuple(attempts)
        self.initial_routing = initial_routing
        self.routing_decision = (
            initial_routing.decision if initial_routing is not None else routing_decision
        )
        self.total_attempts = len(self.attempts)
        self.totals = _aggregate(self.attempts)


class EscalationPipeline:
    """Route once and escalate a failed small-model answer at most once.

    Accept a passing answer first. If both fail, prefer the higher quality score;
    ties retain the initial answer. These heuristics do not verify facts.
    """

    def __init__(self, router, *, evaluation_config=DEFAULT_EVALUATION_CONFIG):
        if not isinstance(router, ModelRouter):
            raise ValueError("router must be a ModelRouter instance.")
        if not isinstance(evaluation_config, EvaluationConfig):
            raise ValueError("evaluation_config must be an EvaluationConfig instance.")
        self.router = router
        self.evaluation_config = evaluation_config

    def _evaluate(self, query, attempts, initial_routing, stage):
        try:
            result = evaluate_answer(
                query, attempts[-1].response.text, config=self.evaluation_config,
            )
            if (
                not isinstance(result, EvaluationResult)
                or result.decision not in ("pass", "fail")
                or isinstance(result.score, bool) or not isinstance(result.score, Real)
                or not math.isfinite(result.score) or not 0 <= result.score <= 1
            ):
                raise ValueError
        except Exception:
            raise EscalationError(
                stage=stage, attempts=attempts, initial_routing=initial_routing,
            ) from None
        return attempts[:-1] + (replace(attempts[-1], evaluation=result),)

    def generate(self, query, *, generation_config=None):
        """Return a completed result or raise EscalationError with available attempts."""
        try:
            initial = self.router.generate(query, generation_config=generation_config)
        except ClassificationError:
            raise EscalationError(stage="classification") from None
        except RoutingGenerationError as error:
            decision = error.decision
            role = decision.selected_role
            model = (self.router.models.config.small_model if role == "small"
                     else self.router.models.config.large_model)
            raise EscalationError(
                stage="initial_generation",
                attempts=(GenerationAttempt(role, model, None, None),),
                routing_decision=decision,
            ) from None

        attempts = (GenerationAttempt(
            initial.response.role, initial.response.model, initial.response, None,
        ),)
        attempts = self._evaluate(query, attempts, initial, "initial_evaluation")
        first = attempts[0]
        large_model = self.router.models.config.large_model
        if first.evaluation.decision == "pass":
            reason = "Initial answer passed evaluation; escalation was not required."
        elif first.role == "large":
            reason = "Initial answer failed evaluation; the large role was already used."
        elif first.model == large_model:
            reason = "Initial answer failed evaluation; both roles use the same model ID."
        else:
            # No rerouting, reclassification, retries, or feedback-prompt rewriting.
            attempts += (GenerationAttempt("large", large_model, None, None),)
            try:
                response = self.router.models.generate(
                    query, role="large", generation_config=generation_config,
                )
                if (
                    not isinstance(response, ModelResponse)
                    or response.role != "large" or response.model != large_model
                ):
                    raise ValueError
            except Exception:
                raise EscalationError(
                    stage="escalation_generation", attempts=attempts, initial_routing=initial,
                ) from None
            attempts = attempts[:-1] + (replace(attempts[-1], response=response),)
            attempts = self._evaluate(query, attempts, initial, "escalation_evaluation")
            reason = "Initial small-model answer failed evaluation; escalated once to the large role."

        escalated = len(attempts) == 2
        final_index = 0
        if escalated:
            second = attempts[1].evaluation
            if second.decision == "pass" or second.score > first.evaluation.score:
                final_index = 1
        return EscalationResult(
            query=query, initial_routing=initial, attempts=attempts,
            escalated=escalated, escalation_reason=reason,
            escalation_role="large" if escalated else None,
            final_attempt_index=final_index, totals=_aggregate(attempts),
        )
