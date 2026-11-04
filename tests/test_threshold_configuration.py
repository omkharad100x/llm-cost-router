"""Offline threshold configuration tests with real SQLite and fake embeddings."""

from unittest.mock import Mock
import numpy as np
import pytest

from cache_config import CacheConfig, DEFAULT_CONFIG
from embeddings import DIMENSIONS, EmbeddingGenerator
from semantic_cache import QueryEligibility, ResponseEligibility, SemanticCache
from semantic_similarity import find_most_similar


QUERY = "What is a tuple?"
ANSWER = "An immutable sequence."
REVIEW = QueryEligibility(
    query=QUERY, context_independent=True, non_sensitive=True,
    time_independent=True, external_state_independent=True,
)
RESPONSE_REVIEW = ResponseEligibility(
    response=ANSWER, successful=True, not_safety_refusal=True, non_sensitive=True,
)


def vector(x, y):
    return np.array([x, y] + [0.0] * (DIMENSIONS - 2))


@pytest.fixture
def cache_inputs(tmp_path):
    embedder = Mock(spec=EmbeddingGenerator)
    embedder.embed_query.return_value = vector(1, 0)
    return tmp_path / "threshold.sqlite3", embedder


def seed(cache):
    assert cache.save(
        QUERY, ANSWER, "test-model",
        eligibility=REVIEW, response_eligibility=RESPONSE_REVIEW,
    ) is not None


def test_default_is_shared_and_preserves_existing_behavior(cache_inputs):
    assert DEFAULT_CONFIG.similarity_threshold == 0.9
    path, embedder = cache_inputs
    with SemanticCache(path, embedder) as cache:
        assert cache.threshold == 0.9
        seed(cache)
        for x, y, expected in [(0.8, 0.6, False), (0.96, 0.28, True)]:
            embedder.embed_query.return_value = vector(x, y)
            result = cache.lookup(QUERY, eligibility=REVIEW)
            match = find_most_similar(vector(x, y), {QUERY: vector(1, 0)})
            assert result.hit is expected
            assert (match is not None) is expected


def test_config_propagation_and_override_precedence(cache_inputs):
    path, embedder = cache_inputs
    config = CacheConfig(similarity_threshold=0.75)
    with SemanticCache(path, embedder, config=config) as cache:
        seed(cache)
        embedder.embed_query.return_value = vector(0.8, 0.6)
        assert cache.lookup(QUERY, eligibility=REVIEW).hit
        assert not cache.lookup(QUERY, eligibility=REVIEW, threshold=0.85).hit
        assert cache.lookup(QUERY, eligibility=REVIEW).hit
        assert cache.threshold == config.similarity_threshold == 0.75
    with SemanticCache(path, embedder, config=config, threshold=0.95) as cache:
        assert not cache.lookup(QUERY, eligibility=REVIEW).hit
        assert cache.lookup(QUERY, eligibility=REVIEW, threshold=0.75).hit
        assert cache.threshold == 0.95


@pytest.mark.parametrize("threshold,x,y", [
    (-1, -1, 0), (0, 0, 1), (1, 1, 0),
    (np.float32(0.75), 0.8, 0.6), (np.int64(0), 0, 1),
])
def test_valid_boundaries_and_numpy_numbers(cache_inputs, threshold, x, y):
    path, embedder = cache_inputs
    with SemanticCache(path, embedder, config=CacheConfig(threshold)) as cache:
        seed(cache)
        embedder.embed_query.return_value = vector(x, y)
        assert cache.lookup(QUERY, eligibility=REVIEW).hit
        assert cache.lookup(QUERY, eligibility=REVIEW, threshold=threshold).hit


@pytest.mark.parametrize("offset,expected", [(-1, False), (0, True), (1, True)])
def test_scores_just_below_equal_and_above_threshold(cache_inputs, offset, expected):
    path, embedder = cache_inputs
    with SemanticCache(path, embedder) as cache:
        seed(cache)
        embedder.embed_query.return_value = vector(0.8, 0.6)
        score = cache.lookup(QUERY, eligibility=REVIEW, threshold=-1).similarity_score
        # Adjacent representable thresholds avoid assumptions about SQLite
        # float32 roundoff and test strict comparison with the measured score.
        threshold = score if offset == 0 else np.nextafter(
            score, np.inf if offset < 0 else -np.inf,
        )
        assert cache.lookup(QUERY, eligibility=REVIEW, threshold=threshold).hit is expected


@pytest.mark.parametrize("invalid", [
    None, True, False, np.bool_(True), "0.9", "", [], {},
    float("nan"), float("inf"), float("-inf"),
    -1.01, 1.01, np.nextafter(-1.0, -np.inf), np.nextafter(1.0, np.inf),
    1 + 0j, 10 ** 1000,
])
def test_invalid_thresholds_fail_before_side_effects(cache_inputs, invalid):
    path, embedder = cache_inputs
    with pytest.raises(ValueError, match="threshold"):
        CacheConfig(invalid)
    with pytest.raises(ValueError, match="threshold"):
        SemanticCache(path, embedder, threshold=invalid)
    assert not path.exists()
    with pytest.raises(ValueError, match="threshold"):
        find_most_similar(vector(1, 0), {}, threshold=invalid)
    with SemanticCache(path, embedder) as cache:
        for eligibility in (None, REVIEW):
            with pytest.raises(ValueError, match="threshold"):
                cache.lookup(QUERY, eligibility=eligibility, threshold=invalid)
        assert cache.threshold == 0.9
    embedder.embed_query.assert_not_called()


def test_override_does_not_bypass_eligibility(cache_inputs):
    path, embedder = cache_inputs
    with SemanticCache(path, embedder) as cache:
        result = cache.lookup(QUERY, threshold=-1)
        assert not result.hit and result.reason == "ineligible"
    embedder.embed_query.assert_not_called()


def test_invalid_config_is_not_ignored(cache_inputs):
    path, embedder = cache_inputs
    with pytest.raises(ValueError, match="CacheConfig"):
        SemanticCache(path, embedder, config={"similarity_threshold": 0.75})
    assert not path.exists()
