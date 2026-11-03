"""Pure offline search tests, plus compatibility with mocked embedding output."""

from unittest.mock import Mock

import numpy as np
import pytest
from google.genai import types

from embeddings import DIMENSIONS, EmbeddingGenerator
from semantic_similarity import SimilarityMatch, find_most_similar


def vector(x=1.0, y=0.0):
    result = np.zeros(DIMENSIONS)
    result[:2] = [x, y]
    return result


def test_identical_vectors_match_at_one():
    query = vector(0.6, 0.8)
    assert find_most_similar(query, {"identical": query}, threshold=1) == SimilarityMatch("identical", 1.0)


def test_similar_vectors_return_query_and_score():
    match = find_most_similar(vector(), {"similar": vector(0.8, 0.6)}, threshold=0.75)
    assert match.query == "similar"
    assert match.score == pytest.approx(0.8)


def test_unrelated_and_opposite_vectors():
    assert find_most_similar(vector(), {"unrelated": vector(0, 1)}) is None
    match = find_most_similar(vector(), {"unrelated": vector(0, 1)}, threshold=0)
    assert match.score == 0
    opposite = find_most_similar(vector(), {"opposite": vector(-1, 0)}, threshold=-1)
    assert opposite.score == -1


def test_inclusive_threshold_boundary_without_interior_tolerance():
    candidates = {"similar": vector(0.8, 0.6)}
    score = find_most_similar(vector(), candidates, threshold=-1).score
    assert find_most_similar(vector(), candidates, threshold=score) is not None
    assert find_most_similar(vector(), candidates, threshold=np.nextafter(score, -np.inf)) is not None
    assert find_most_similar(vector(), candidates, threshold=np.nextafter(score, np.inf)) is None


def test_empty_collection():
    assert find_most_similar(vector(), {}) is None


def test_best_of_multiple_candidates_and_first_tie():
    candidates = {
        "unrelated": vector(0, 1),
        "good": vector(0.8, 0.6),
        "best": vector(),
        "equal best": vector(),
    }
    assert find_most_similar(vector(), candidates) == SimilarityMatch("best", 1.0)


@pytest.mark.parametrize("invalid", [
    [], [1, 0], [[1, 0]], [0] * DIMENSIONS,
    [float("nan")] * DIMENSIONS, [float("inf")] * DIMENSIONS,
    ["0"] * DIMENSIONS, [True] * DIMENSIONS,
    [1j] * DIMENSIONS, None, 1,
    [[1], [0, 1]], vector(2, 0), vector(1e308, 1e308),
])
def test_invalid_candidate_is_rejected(invalid):
    with pytest.raises(ValueError, match="Vectors must"):
        find_most_similar(vector(), {"invalid": invalid})


@pytest.mark.parametrize("invalid", [None, [], [[1]], [float("nan")], [0, 0], [2, 0]])
def test_invalid_query_is_rejected_even_without_candidates(invalid):
    with pytest.raises(ValueError, match="Vectors must"):
        find_most_similar(invalid, {})


@pytest.mark.parametrize("threshold", [None, True, np.bool_(False), "0.9", float("nan"), float("inf"), -1.1, 1.1, 1 + 0j, 10 ** 1000])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        find_most_similar(vector(), {}, threshold=threshold)


@pytest.mark.parametrize("collection", [None, [], [("query", vector())], {"": vector()}, {1: vector()}])
def test_invalid_collection_or_query_text(collection):
    with pytest.raises(ValueError):
        find_most_similar(vector(), collection)


def test_inputs_are_not_modified_and_float32_drift_is_accepted():
    query = vector(0.6, 0.8).astype(np.float32)
    candidate = query.copy()
    original = query.copy()
    match = find_most_similar(query, {"query": candidate}, threshold=1)
    assert match.score == 1
    np.testing.assert_array_equal(query, original)
    np.testing.assert_array_equal(candidate, original)


def test_embedding_module_output_can_be_used_directly():
    generator = Mock()
    generator.embed_content.return_value = types.EmbedContentResponse(
        embeddings=[types.ContentEmbedding(values=vector(3, 4).tolist())]
    )
    query = EmbeddingGenerator(generator).embed_query("Synthetic query")
    match = find_most_similar(query, {"stored query": query}, threshold=1)
    assert match == SimilarityMatch("stored query", 1.0)
    # Only embedding creation uses the mocked provider; search does not.
    generator.embed_content.assert_called_once()


def test_different_nonempty_dimension_supported_when_consistent():
    assert find_most_similar([1, 0], {"small fixture": [1, 0]}, threshold=1).score == 1
    with pytest.raises(ValueError, match="matching dimensions"):
        find_most_similar([1, 0], {"wrong dimensions": [1, 0, 0]})
