"""Validated settings for the deterministic query complexity classifier."""

from dataclasses import dataclass


DEFAULT_WEIGHTS = (
    ("length", 1), ("multiple_tasks", 3), ("multi_step", 3),
    ("reasoning", 3), ("coding", 3), ("advanced_math", 3),
    ("analysis", 3), ("explanation", 1), ("context", 1),
)


@dataclass(frozen=True)
class ComplexityConfig:
    """A rule contributes its weight at most once; zero disables its weight."""

    complex_threshold: int = 3
    long_query_words: int = 120
    multiple_tasks_min: int = 2
    weights: tuple[tuple[str, int], ...] = DEFAULT_WEIGHTS

    def __post_init__(self):
        for name in ("complex_threshold", "long_query_words", "multiple_tasks_min"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.multiple_tasks_min < 2:
            raise ValueError("multiple_tasks_min must be at least 2.")
        if (
            type(self.weights) is not tuple
            or any(type(item) is not tuple or len(item) != 2 for item in self.weights)
        ):
            raise ValueError("weights must be an immutable tuple of (signal, weight) pairs.")
        names = [name for name, _ in self.weights]
        required = {name for name, _ in DEFAULT_WEIGHTS}
        if any(type(name) is not str for name in names) or len(names) != len(set(names)):
            raise ValueError("Signal names must be unique strings.")
        if set(names) != required:
            raise ValueError("weights must define exactly the supported signal names.")
        if any(type(weight) is not int or weight < 0 for _, weight in self.weights):
            raise ValueError("Signal weights must be nonnegative integers.")


DEFAULT_COMPLEXITY_CONFIG = ComplexityConfig()
