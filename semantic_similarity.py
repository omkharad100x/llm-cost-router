"""In-memory cosine search over normalized query embeddings.

Pass vectors returned by embeddings.EmbeddingGenerator.embed_query. All vectors
must use the same embedding model, task type, and dimensionality. No API or
storage access occurs here.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from cache_config import DEFAULT_CONFIG, _UNSET, validate_threshold


@dataclass(frozen=True)
class SimilarityMatch:
    """The stored query and cosine score of an accepted match."""

    query: str
    score: float


def _normalized_vector(value, dimensions=None):
    """Validate a vector and remove harmless floating-point norm drift."""
    try:
        vector = np.asarray(value)
        if (
            vector.ndim != 1
            or vector.size == 0
            or vector.dtype.kind not in "fiu"
            or (dimensions is not None and vector.size != dimensions)
        ):
            raise ValueError
        vector = vector.astype(np.float64)
        if not np.isfinite(vector).all():
            raise ValueError
        # A unit vector cannot contain a component greater than one. Reject
        # grossly invalid data before computing a potentially overflowing norm.
        if np.max(np.abs(vector)) > 1.000001:
            raise ValueError
        norm = np.linalg.norm(vector)
        if not np.isclose(norm, 1.0, rtol=1e-6, atol=1e-8):
            raise ValueError
        return vector / norm
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            "Vectors must be finite, nonempty, normalized numeric arrays "
            "with matching dimensions."
        ) from None


def find_most_similar(query_vector, stored_queries, threshold=_UNSET):
    """Return SimilarityMatch or None if no candidate reaches the threshold.

    stored_queries maps nonempty query strings to normalized vectors. Scores
    range from -1 to 1; the threshold is inclusive and defaults to 0.9.
    Equal scores select the first mapping entry. Invalid inputs raise ValueError,
    even when the collection is empty. Vectors are copied, never modified.
    """
    threshold = validate_threshold(
        DEFAULT_CONFIG.similarity_threshold if threshold is _UNSET else threshold
    )
    if not isinstance(stored_queries, Mapping):
        raise ValueError("Stored queries must be a mapping of query text to vectors.")
    query = _normalized_vector(query_vector)
    if not stored_queries:
        return None

    texts = []
    vectors = []
    for text, vector in stored_queries.items():
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Stored queries must have nonempty text keys.")
        texts.append(text)
        vectors.append(_normalized_vector(vector, dimensions=query.size))

    scores = np.clip(np.stack(vectors) @ query, -1.0, 1.0)
    # Stabilize the endpoints against machine roundoff for identical/opposite
    # vectors. Interior scores use a strict, inclusive floating-point comparison.
    endpoint_error = 8 * np.finfo(np.float64).eps
    scores[np.abs(scores - 1.0) <= endpoint_error] = 1.0
    scores[np.abs(scores + 1.0) <= endpoint_error] = -1.0
    index = int(np.argmax(scores))
    score = float(scores[index])
    if score < threshold:
        return None
    return SimilarityMatch(query=texts[index], score=score)
