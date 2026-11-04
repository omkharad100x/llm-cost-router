"""Offline persistent-cache tests using temporary SQLite files and fake embeddings."""

from dataclasses import replace
from unittest.mock import Mock
import sqlite3

import numpy as np
import pytest

from embeddings import DIMENSIONS, EmbeddingGenerator, MODEL
from instrumentation import InstrumentedGenerator
from semantic_cache import (
    CacheConflictError, CacheError, QueryEligibility, ResponseEligibility,
    SemanticCache,
)
from google.genai import types


def vector(x=1.0, y=0.0):
    result = np.zeros(DIMENSIONS)
    result[:2] = [x, y]
    return result


def approved(query):
    # Test fixtures are synthetic and explicitly reviewed by this test.
    return QueryEligibility(
        query=query, context_independent=True, non_sensitive=True,
        time_independent=True, external_state_independent=True,
    )


def approved_response(response):
    return ResponseEligibility(
        response=response, successful=True, not_safety_refusal=True,
        non_sensitive=True,
    )


def save(cache, query="What is a tuple?", answer="An immutable sequence.", model="test-model", **kwargs):
    return cache.save(
        query, answer, model, eligibility=approved(query),
        response_eligibility=approved_response(answer), **kwargs,
    )


def lookup(cache, query="What is a tuple?"):
    return cache.lookup(query, eligibility=approved(query))


@pytest.fixture
def setup(tmp_path):
    path = tmp_path / "cache.sqlite3"
    now = [1000.0]
    embedder = Mock(spec=EmbeddingGenerator)
    embedder.embed_query.return_value = vector()
    with SemanticCache(path, embedder, ttl_seconds=60, clock=lambda: now[0]) as cache:
        yield cache, embedder, path, now


def rows(path):
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        return connection.execute("SELECT * FROM cache_entries").fetchall()


def test_database_initialization_and_empty_miss(setup):
    cache, embedder, path, _ = setup
    assert path.exists()
    assert rows(path) == []
    result = lookup(cache)
    assert not result.hit
    assert result.response is result.matched_query is result.similarity_score is result.entry_id is None
    embedder.embed_query.assert_called_once()
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2


def test_exact_hit_and_schema_storage(setup):
    cache, _, path, _ = setup
    entry_id = save(cache)
    result = lookup(cache)
    assert result.hit
    assert result.entry_id == entry_id
    assert result.response == "An immutable sequence."
    assert result.matched_query == "What is a tuple?"
    assert result.similarity_score == 1
    row = rows(path)[0]
    assert row["response_model"] == "test-model"
    assert row["created_at"] == 1000
    assert row["expires_at"] == 1060
    assert row["embedding_model"] == MODEL
    assert row["embedding_version"] == "v1"
    assert row["embedding_task"] == "SEMANTIC_SIMILARITY"
    assert row["embedding_dimensions"] == DIMENSIONS
    assert len(row["embedding"]) == DIMENSIONS * 4
    assert np.linalg.norm(np.frombuffer(row["embedding"], dtype="<f4")) == pytest.approx(1)


def test_similar_hit_and_unrelated_miss(setup):
    cache, embedder, _, _ = setup
    save(cache)
    embedder.embed_query.return_value = vector(0.96, 0.28)
    result = lookup(cache, "Explain a tuple.")
    assert result.hit and result.similarity_score == pytest.approx(0.96)
    embedder.embed_query.return_value = vector(0, 1)
    assert not lookup(cache, "Describe photosynthesis.").hit


def test_configurable_threshold_and_best_candidate(setup):
    cache, embedder, path, now = setup
    save(cache, "first")
    embedder.embed_query.return_value = vector(0.8, 0.6)
    best_id = save(cache, "second", "Second answer")
    assert lookup(cache, "third").entry_id == best_id
    with SemanticCache(path, embedder, threshold=1, clock=lambda: now[0]) as strict:
        embedder.embed_query.return_value = vector(0, 1)
        assert not lookup(strict, "third").hit
        embedder.embed_query.return_value = vector(0.8, 0.6)
        # Stored float32 vectors are very close; use the exact persisted vector.
        embedder.embed_query.return_value = np.frombuffer(
            rows(path)[1]["embedding"], dtype="<f4"
        ).astype(float)
        assert lookup(strict, "third").hit


def test_threshold_equality_at_zero(setup):
    _, embedder, path, now = setup
    with SemanticCache(path, embedder, threshold=0, clock=lambda: now[0]) as cache:
        save(cache)
        embedder.embed_query.return_value = vector(0, 1)
        assert lookup(cache, "orthogonal").similarity_score == 0


def test_default_ttl_boundary_cleanup_and_override(setup):
    cache, _, path, now = setup
    save(cache)
    now[0] = 1059.999
    assert lookup(cache).hit
    now[0] = 1060
    assert not lookup(cache).hit
    assert cache.remove_expired() == 1
    assert cache.remove_expired() == 0
    save(cache, ttl_seconds=5)
    assert rows(path)[0]["expires_at"] == 1065
    now[0] = 1065
    assert not lookup(cache).hit


def test_expiration_checked_after_embedding_latency(setup):
    cache, embedder, _, now = setup
    save(cache, ttl_seconds=1)

    def delayed(_):
        now[0] += 2
        return vector()

    embedder.embed_query.side_effect = delayed
    assert not lookup(cache).hit


def test_duplicate_reuses_id_without_call_or_ttl_refresh(setup):
    cache, embedder, path, now = setup
    first = save(cache)
    calls = embedder.embed_query.call_count
    now[0] += 20
    assert save(cache, ttl_seconds=999) == first
    assert embedder.embed_query.call_count == calls
    assert len(rows(path)) == 1
    assert rows(path)[0]["expires_at"] == 1060


@pytest.mark.parametrize("answer,model", [
    ("Conflicting answer", "test-model"),
    ("An immutable sequence.", "different-model"),
])
def test_active_conflict_rejected(setup, answer, model):
    cache, embedder, path, _ = setup
    first = save(cache)
    calls = embedder.embed_query.call_count
    with pytest.raises(CacheConflictError):
        save(cache, answer=answer, model=model)
    assert rows(path)[0]["id"] == first
    assert embedder.embed_query.call_count == calls


def test_expired_duplicate_is_replaced(setup):
    cache, _, path, now = setup
    first = save(cache)
    now[0] = 1060
    second = save(cache, answer="New approved response")
    assert first != second
    assert len(rows(path)) == 1
    assert lookup(cache).response == "New approved response"


def test_duplicate_key_strips_edges_only_preserves_original(setup):
    cache, _, path, _ = setup
    first = save(cache, query="  What is a tuple?  ")
    assert save(cache) == first
    assert rows(path)[0]["original_query"] == "  What is a tuple?  "


@pytest.mark.parametrize("field", [
    "context_independent", "non_sensitive", "time_independent",
    "external_state_independent",
])
def test_ineligible_queries_never_embed_or_store(setup, field):
    cache, embedder, path, _ = setup
    query = "Synthetic explicitly disallowed input"
    decision = replace(approved(query), **{field: False})
    result = cache.lookup(query, eligibility=decision)
    assert not result.hit and result.reason == "ineligible"
    assert cache.save(query, "answer", "model", eligibility=decision,
                      response_eligibility=approved_response("answer")) is None
    embedder.embed_query.assert_not_called()
    assert rows(path) == []


@pytest.mark.parametrize("decision", [None, QueryEligibility(query="query"), True, {"allowed": True}])
def test_default_deny_and_no_implicit_standalone_approval(setup, decision):
    cache, embedder, _, _ = setup
    assert not cache.lookup("query", eligibility=decision).hit
    assert cache.save("query", "answer", "model", eligibility=decision,
                      response_eligibility=approved_response("answer")) is None
    embedder.embed_query.assert_not_called()


@pytest.mark.parametrize("field", ["successful", "not_safety_refusal", "non_sensitive"])
def test_failed_refused_or_sensitive_responses_not_cached(setup, field):
    cache, embedder, path, _ = setup
    decision = replace(approved_response("answer"), **{field: False})
    assert cache.save("query", "answer", "model", eligibility=approved("query"),
                      response_eligibility=decision) is None
    embedder.embed_query.assert_not_called()
    assert not rows(path)


def test_eligibility_is_bound_to_exact_content_and_boolean_flags(setup):
    cache, embedder, _, _ = setup
    assert not cache.lookup("other", eligibility=approved("query")).hit
    assert not cache.lookup("query", eligibility=replace(approved("query"), non_sensitive=1)).hit
    assert cache.save("query", "changed answer", "model", eligibility=approved("query"),
                      response_eligibility=approved_response("original answer")) is None
    assert cache.save("query", "answer", "model", eligibility=approved("query")) is None
    embedder.embed_query.assert_not_called()


@pytest.mark.parametrize("empty", ["", " \t", None])
def test_empty_requests_and_responses_not_cached(setup, empty):
    cache, embedder, _, _ = setup
    assert not cache.lookup(empty, eligibility=approved(empty)).hit
    assert cache.save("query", empty, "model", eligibility=approved("query"),
                      response_eligibility=approved_response(empty)) is None
    embedder.embed_query.assert_not_called()


@pytest.mark.parametrize("bad", [
    [], [1, 0], np.zeros(DIMENSIONS),
    np.full(DIMENSIONS, np.nan), vector(2, 0),
])
def test_invalid_generated_embedding_not_stored_or_searched(setup, bad):
    cache, embedder, path, _ = setup
    embedder.embed_query.return_value = bad
    with pytest.raises(ValueError):
        save(cache)
    with pytest.raises(ValueError):
        lookup(cache)
    assert not rows(path)


@pytest.mark.parametrize("blob", [
    b"short", np.zeros(DIMENSIONS, dtype="<f4").tobytes(),
    np.full(DIMENSIONS, np.nan, dtype="<f4").tobytes(),
    np.full(DIMENSIONS, np.inf, dtype="<f4").tobytes(),
    vector(2, 0).astype("<f4").tobytes(), "not a blob",
])
def test_corrupt_embeddings_skipped_without_hiding_valid_candidate(setup, blob):
    cache, _, path, _ = setup
    damaged_id = save(cache, "damaged")
    good_id = save(cache, "good")
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE cache_entries SET embedding = ? WHERE id = ?", (blob, damaged_id))
    assert lookup(cache, "search").entry_id == good_id


@pytest.mark.parametrize("field,value", [
    ("embedding_model", "different"), ("embedding_version", "different"),
    ("embedding_task", "RETRIEVAL_QUERY"), ("embedding_dimensions", 3),
    ("embedding_dtype", ">f4"), ("eligibility_version", "old-policy"),
])
def test_incompatible_embeddings_are_not_searched(setup, field, value):
    cache, _, path, _ = setup
    entry_id = save(cache)
    # Field names are fixed test parameters, never untrusted application input.
    with sqlite3.connect(path) as connection:
        connection.execute(f"UPDATE cache_entries SET {field} = ? WHERE id = ?", (value, entry_id))
    assert not lookup(cache).hit


def test_corrupt_exact_entry_can_be_rebuilt(setup):
    cache, _, path, _ = setup
    first = save(cache)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE cache_entries SET embedding = ? WHERE id = ?", (b"bad", first))
    second = save(cache)
    assert second != first
    assert len(rows(path)) == 1
    assert lookup(cache).hit


def test_clear_namespace_and_persistence(tmp_path):
    path = tmp_path / "persistent.db"
    embedder = Mock(spec=EmbeddingGenerator)
    embedder.embed_query.return_value = vector()
    with SemanticCache(path, embedder, namespace="first") as first:
        first_id = save(first)
    with SemanticCache(path, embedder, namespace="second") as second:
        assert not lookup(second).hit
        second_id = save(second)
    with SemanticCache(path, embedder, namespace="first") as reopened:
        assert lookup(reopened).entry_id == first_id
        assert reopened.clear() == 1
        assert reopened.clear() == 0
        assert not lookup(reopened).hit
    with SemanticCache(path, embedder, namespace="second") as second:
        assert lookup(second).entry_id == second_id


def test_cleanup_preserves_other_namespaces(setup):
    cache, embedder, path, now = setup
    save(cache)
    with SemanticCache(path, embedder, namespace="other", clock=lambda: now[0]) as other:
        save(other)
    now[0] += 61
    assert cache.remove_expired() == 1
    assert len(rows(path)) == 1


def test_parameterized_sql_preserves_quotes_and_sql_like_text(setup):
    cache, _, path, _ = setup
    query = "Explain '; DROP TABLE cache_entries; --"
    answer = "Quotes: ' and \""
    entry_id = save(cache, query, answer)
    assert lookup(cache, query).entry_id == entry_id
    assert rows(path)[0]["cached_response"] == answer


def test_concurrent_conflict_rechecked_after_embedding(setup):
    cache, embedder, path, now = setup
    other_embedder = Mock(spec=EmbeddingGenerator)
    other_embedder.embed_query.return_value = vector()
    with SemanticCache(path, other_embedder, clock=lambda: now[0]) as other:
        def concurrent_write(_):
            save(other, answer="Other answer")
            return vector()
        embedder.embed_query.side_effect = concurrent_write
        with pytest.raises(CacheConflictError):
            save(cache)
    assert len(rows(path)) == 1
    assert rows(path)[0]["cached_response"] == "Other answer"


def test_embedding_failure_does_not_mutate_database(setup):
    cache, embedder, path, _ = setup
    error = RuntimeError("fake API failure")
    embedder.embed_query.side_effect = error
    with pytest.raises(RuntimeError) as caught:
        save(cache)
    assert caught.value is error
    assert not rows(path)


@pytest.mark.parametrize("ttl", [0, -1, True, float("inf"), float("nan"), "60"])
def test_invalid_ttl(setup, ttl):
    cache, embedder, path, _ = setup
    with pytest.raises(ValueError):
        save(cache, ttl_seconds=ttl)
    with pytest.raises(ValueError):
        SemanticCache(path, embedder, ttl_seconds=ttl)


def test_rejects_unrelated_corrupt_or_future_database(tmp_path):
    embedder = Mock()
    foreign = tmp_path / "foreign.db"
    with sqlite3.connect(foreign) as connection:
        connection.execute("CREATE TABLE unrelated (value TEXT)")
    with pytest.raises(CacheError):
        SemanticCache(foreign, embedder)
    with sqlite3.connect(foreign) as connection:
        assert connection.execute("SELECT * FROM unrelated").fetchall() == []
    corrupt = tmp_path / "corrupt.db"
    corrupt.write_bytes(b"this is not sqlite")
    with pytest.raises(CacheError):
        SemanticCache(corrupt, embedder)
    future = tmp_path / "future.db"
    with sqlite3.connect(future) as connection:
        connection.execute("PRAGMA user_version = 99")
    with pytest.raises(CacheError):
        SemanticCache(future, embedder)


def test_closed_cache_raises_clear_error(setup):
    cache, embedder, _, _ = setup
    cache.close()
    with pytest.raises(CacheError, match="closed"):
        lookup(cache)
    embedder.embed_query.assert_not_called()


def test_real_embedding_and_instrumentation_wiring_with_mocked_sdk(tmp_path):
    client = Mock()
    client.models.embed_content.return_value = types.EmbedContentResponse(
        embeddings=[types.ContentEmbedding(values=vector().tolist())]
    )
    records = []
    embedder = EmbeddingGenerator(InstrumentedGenerator(client, {}, records.append))
    with SemanticCache(tmp_path / "wired.db", embedder) as cache:
        save(cache)
        assert lookup(cache).hit
    assert [record["request_kind"] for record in records] == ["embedding", "embedding"]
    client.models.generate_content.assert_not_called()


def test_match_that_expires_during_scan_returns_miss(setup, monkeypatch):
    import semantic_cache
    cache, _, _, now = setup
    save(cache, ttl_seconds=1)
    original = semantic_cache.find_most_similar

    def slow_match(query, candidates, threshold=0.9):
        result = original(query, candidates, threshold)
        if candidates:
            now[0] += 2
        return result

    monkeypatch.setattr(semantic_cache, "find_most_similar", slow_match)
    result = lookup(cache)
    assert not result.hit and result.reason == "expired"


def test_database_operational_errors_are_sanitized(setup):
    cache, _, path, _ = setup
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE cache_entries")
    with pytest.raises(CacheError, match="Cache lookup failed"):
        lookup(cache)
    with pytest.raises(CacheError, match="Cache storage failed"):
        save(cache)


def test_insert_failure_rolls_back_expired_entry_replacement(setup):
    cache, _, path, now = setup
    first = save(cache)
    now[0] += 61
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TRIGGER reject_insert BEFORE INSERT ON cache_entries "
            "BEGIN SELECT RAISE(ABORT, 'synthetic failure'); END"
        )
    with pytest.raises(CacheError, match="Cache storage failed"):
        save(cache)
    assert len(rows(path)) == 1
    assert rows(path)[0]["id"] == first


def test_embedding_version_scopes_entries_after_reopening(setup):
    cache, embedder, path, now = setup
    save(cache)
    with SemanticCache(path, embedder, embedding_version="v2", clock=lambda: now[0]) as v2:
        assert not lookup(v2).hit
        new_id = save(v2)
        assert lookup(v2).entry_id == new_id
    assert len(rows(path)) == 2
