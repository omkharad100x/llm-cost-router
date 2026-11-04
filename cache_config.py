"""Shared, explicit configuration for semantic search; no environment or API access."""

from dataclasses import dataclass
import math
from numbers import Real


def validate_threshold(value):
    """Accept finite real numbers in [-1, 1]; reject booleans and coercion."""
    if (
        isinstance(value, bool) or not isinstance(value, Real)
        or not -1 <= value <= 1 or not math.isfinite(value)
    ):
        raise ValueError("Similarity threshold must be a finite number from -1 to 1.")
    return value


@dataclass(frozen=True)
class CacheConfig:
    """Application defaults; pass a custom instance to SemanticCache."""

    similarity_threshold: float = 0.9
    cache_version: str = "v1"

    def __post_init__(self):
        validate_threshold(self.similarity_threshold)
        if not isinstance(self.cache_version, str) or not self.cache_version.strip():
            raise ValueError("Cache version must be a nonempty string.")


DEFAULT_CONFIG = CacheConfig()
# Omission is distinct from explicitly supplying an invalid None.
_UNSET = object()
