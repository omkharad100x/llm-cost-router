"""Validated, immutable settings for deterministic answer-quality checks."""

from dataclasses import dataclass
import math
from numbers import Real


DEFAULT_WEIGHTS = (
    ("presence", 20), ("relevance", 30), ("completeness", 25),
    ("structure", 10), ("certainty", 10), ("error_free", 5),
)


@dataclass(frozen=True)
class EvaluationConfig:
    acceptance_threshold: float = 0.8
    short_answer_max_words: int = 3
    max_paragraph_chars: int = 1200
    fail_on_uncertainty: bool = True
    fail_on_error: bool = True
    fail_on_incomplete: bool = True
    weights: tuple[tuple[str, int], ...] = DEFAULT_WEIGHTS

    def __post_init__(self):
        value = self.acceptance_threshold
        if (
            isinstance(value, bool) or not isinstance(value, Real)
            or not 0 <= value <= 1 or not math.isfinite(value)
        ):
            raise ValueError("acceptance_threshold must be finite and in [0, 1].")
        for name in ("short_answer_max_words", "max_paragraph_chars"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("fail_on_uncertainty", "fail_on_error", "fail_on_incomplete"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean.")
        if type(self.weights) is not tuple or any(
            type(item) is not tuple or len(item) != 2 for item in self.weights
        ):
            raise ValueError("weights must be an immutable tuple of pairs.")
        names = [name for name, _ in self.weights]
        if any(type(name) is not str for name in names):
            raise ValueError("Criterion names must be strings.")
        if len(names) != len(set(names)) or set(names) != {name for name, _ in DEFAULT_WEIGHTS}:
            raise ValueError("weights must define each supported criterion exactly once.")
        if any(type(weight) is not int or weight < 0 for _, weight in self.weights):
            raise ValueError("Weights must be nonnegative integers.")
        if not sum(weight for _, weight in self.weights):
            raise ValueError("At least one criterion must have positive weight.")


DEFAULT_EVALUATION_CONFIG = EvaluationConfig()
