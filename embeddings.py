"""Generate query embeddings using an existing instrumented Gemini client.

Importing this module never loads configuration, reads secrets, or calls an API.
"""

import numpy as np
from google.genai import types


MODEL = "gemini-embedding-001"
DIMENSIONS = 768


class EmbeddingResponseError(ValueError):
    """The provider returned no usable embedding for the complete query."""


class EmbeddingGenerator:
    """Embed one query at a time; inject an InstrumentedGenerator instance."""

    def __init__(self, generator):
        self.generator = generator

    def embed_query(self, text):
        """Return a finite, unit-length float64 array of shape (768,).

        Invalid local input raises ValueError before a request. SDK exceptions
        propagate unchanged; invalid provider output raises EmbeddingResponseError.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Embedding input must be a nonempty text string.")

        response = self.generator.embed_content(
            model=MODEL,
            contents=text.strip(),
            config=types.EmbedContentConfig(
                task_type="SEMANTIC_SIMILARITY",
                output_dimensionality=DIMENSIONS,
            ),
        )
        embeddings = getattr(response, "embeddings", None)
        if not isinstance(embeddings, (list, tuple)) or len(embeddings) != 1:
            raise EmbeddingResponseError("Expected exactly one query embedding.")

        embedding = embeddings[0]
        statistics = getattr(embedding, "statistics", None)
        if getattr(statistics, "truncated", False):
            raise EmbeddingResponseError("The provider truncated the query.")

        values = getattr(embedding, "values", None)
        try:
            vector = np.asarray(values)
            if vector.shape != (DIMENSIONS,) or vector.dtype.kind not in "fiu":
                raise ValueError
            vector = vector.astype(np.float64)
            if not np.isfinite(vector).all():
                raise ValueError
            # Scale first to avoid overflow or underflow in the norm.
            scale = np.max(np.abs(vector))
            if scale == 0:
                raise ValueError
            scaled = vector / scale
            return scaled / np.linalg.norm(scaled)
        except (ValueError, TypeError, OverflowError):
            raise EmbeddingResponseError(
                "Expected a finite, nonzero numeric vector with 768 dimensions."
            ) from None
