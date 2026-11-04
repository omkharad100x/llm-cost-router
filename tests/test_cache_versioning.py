"""SQLite migration, version isolation, and targeted invalidation tests."""

from contextlib import closing
from unittest.mock import Mock
import sqlite3

import numpy as np
import pytest

import cache_schema
import semantic_cache
from cache_config import CacheConfig
from cache_schema import LEGACY_SCHEMA, LEGACY_EMBEDDING_CONFIG
from embeddings import DIMENSIONS, MODEL, EmbeddingGenerator
from semantic_cache import CacheError, QueryEligibility, ResponseEligibility, SemanticCache


QUERY = "What is a tuple?"
ANSWER = "An immutable sequence."
REVIEW = QueryEligibility(
    query=QUERY, context_independent=True, non_sensitive=True,
    time_independent=True, external_state_independent=True,
)
RESPONSE = ResponseEligibility(
    response=ANSWER, successful=True, not_safety_refusal=True, non_sensitive=True,
)


@pytest.fixture
def inputs(tmp_path):
    embedder = Mock(spec=EmbeddingGenerator)
    vector = np.zeros(DIMENSIONS)
    vector[0] = 1
    embedder.embed_query.return_value = vector
    return tmp_path / "versions.sqlite3", embedder


def save(cache, **kwargs):
    return cache.save(QUERY, ANSWER, "test-model", eligibility=REVIEW,
                      response_eligibility=RESPONSE, **kwargs)


def lookup(cache):
    return cache.lookup(QUERY, eligibility=REVIEW)


def inspect(path):
    with closing(sqlite3.connect(path)) as db:
        db.row_factory = sqlite3.Row
        return db.execute("PRAGMA user_version").fetchone()[0], [
            dict(row) for row in db.execute("SELECT * FROM cache_entries ORDER BY id")
        ]


def legacy_database(path, embedder):
    with closing(sqlite3.connect(path)) as db, db:
        db.execute(LEGACY_SCHEMA)
        db.execute("CREATE INDEX cache_expiration ON cache_entries(namespace, expires_at)")
        for entry_id, namespace, expires in [("live", "default", 2000), ("expired", "other", 500)]:
            db.execute(
                "INSERT INTO cache_entries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (entry_id, namespace, QUERY, QUERY, ANSWER, "test-model",
                 embedder.embed_query.return_value.astype("<f4").tobytes(),
                 MODEL, "v1", "SEMANTIC_SIMILARITY", DIMENSIONS, "<f4",
                 "explicit-review-v1", 100, expires),
            )
        db.execute("PRAGMA user_version = 1")


def test_migration_preserves_every_original_field_and_persistence(inputs):
    path, embedder = inputs
    legacy_database(path, embedder)
    _, before = inspect(path)
    with SemanticCache(path, embedder, clock=lambda: 1000) as cache:
        assert lookup(cache).entry_id == "live"
    version, after = inspect(path)
    assert version == 2 and len(after) == len(before) == 2
    for old, new in zip(before, after):
        assert {key: new[key] for key in old} == old
        assert new["cache_version"] == "v1"
        assert new["embedding_config"] == LEGACY_EMBEDDING_CONFIG
    with SemanticCache(path, embedder, clock=lambda: 1000) as reopened:
        assert lookup(reopened).entry_id == "live"
    assert inspect(path) == (version, after)


def test_migration_does_not_assign_old_data_to_requested_new_version(inputs):
    path, embedder = inputs
    legacy_database(path, embedder)
    with SemanticCache(path, embedder, cache_version="v2", clock=lambda: 1000) as cache:
        assert not lookup(cache).hit
        new_id = save(cache)
        assert lookup(cache).entry_id == new_id
    assert len(inspect(path)[1]) == 3


def test_failure_after_table_rebuild_rolls_back_schema_and_data(inputs, monkeypatch):
    path, embedder = inputs
    legacy_database(path, embedder)
    before = inspect(path)

    def fail(_):
        raise sqlite3.OperationalError("synthetic disk failure")

    monkeypatch.setattr(cache_schema, "create_expiration_index", fail)
    with pytest.raises(CacheError):
        SemanticCache(path, embedder)
    assert inspect(path) == before
    with closing(sqlite3.connect(path)) as db:
        assert db.execute(
            "SELECT name FROM sqlite_master WHERE name = 'cache_entries_v1'"
        ).fetchone() is None


@pytest.mark.parametrize("custom", ["column", "trigger", "table"])
def test_custom_legacy_schema_refused_without_data_loss(inputs, custom):
    path, embedder = inputs
    legacy_database(path, embedder)
    with closing(sqlite3.connect(path)) as db, db:
        if custom == "column":
            db.execute("ALTER TABLE cache_entries ADD COLUMN personal_note TEXT")
        elif custom == "trigger":
            db.execute("CREATE TRIGGER custom AFTER DELETE ON cache_entries BEGIN SELECT 1; END")
        else:
            db.execute("CREATE TABLE user_data (note TEXT)")
            db.execute("INSERT INTO user_data VALUES ('synthetic note')")
    before = inspect(path)
    with pytest.raises(CacheError):
        SemanticCache(path, embedder)
    assert inspect(path) == before


def test_application_versions_coexist_without_conflicts(inputs):
    path, embedder = inputs
    with SemanticCache(path, embedder) as old:
        first = save(old)
        with SemanticCache(path, embedder, config=CacheConfig(cache_version="v2")) as new:
            assert not lookup(new).hit
            second = save(new)
            assert second != first
            assert lookup(new).entry_id == second
            assert lookup(old).entry_id == first
    assert len(inspect(path)[1]) == 2


@pytest.mark.parametrize("field,value", [
    ("embedding_config", '{"normalization":"different"}'),
    ("cache_version", "other"), ("embedding_model", "other"),
    ("embedding_version", "other"), ("embedding_task", "other"),
    ("embedding_dimensions", 3), ("embedding_dtype", ">f4"),
])
def test_incompatible_metadata_never_reused(inputs, field, value):
    path, embedder = inputs
    with SemanticCache(path, embedder) as cache:
        entry_id = save(cache)
        with closing(sqlite3.connect(path)) as db, db:
            db.execute(f"UPDATE cache_entries SET {field} = ? WHERE id = ?", (value, entry_id))
        assert not lookup(cache).hit


@pytest.mark.parametrize("selector", ["entry_id", "namespace", "embedding_model", "embedding_version", "cache_version"])
def test_invalidation_is_scoped_and_persists(inputs, selector):
    path, embedder = inputs
    with SemanticCache(path, embedder) as cache:
        entry_id = save(cache)
        with SemanticCache(path, embedder, namespace="other") as other:
            other_id = save(other)
        values = dict(entry_id=entry_id, namespace="default", embedding_model=MODEL,
                      embedding_version="v1", cache_version="v1")
        assert cache.invalidate(**{selector: values[selector]}) == 1
        assert cache.invalidate(**{selector: values[selector]}) == 0
        assert not lookup(cache).hit
    with SemanticCache(path, embedder, namespace="other") as other:
        assert lookup(other).entry_id == other_id


def test_filters_combine_and_namespace_can_be_explicit(inputs):
    path, embedder = inputs
    with SemanticCache(path, embedder) as cache:
        save(cache)
        with SemanticCache(path, embedder, namespace="other", cache_version="v2") as other:
            other_id = save(other)
        assert cache.invalidate(embedding_model=MODEL, cache_version="absent") == 0
        assert cache.invalidate(entry_id=other_id) == 0
        assert cache.invalidate(namespace="other", cache_version="v2", embedding_model=MODEL) == 1
        assert lookup(cache).hit
        assert cache.invalidate(embedding_model="' OR 1=1 --") == 0


@pytest.mark.parametrize("selector", ["entry_id", "namespace", "embedding_model", "cache_version"])
def test_invalidation_during_search_cannot_return_snapshot(inputs, monkeypatch, selector):
    path, embedder = inputs
    with SemanticCache(path, embedder) as reader, SemanticCache(path, embedder) as writer:
        entry_id = save(reader)
        original = semantic_cache.find_most_similar

        def search(vector, candidates, threshold):
            result = original(vector, candidates, threshold)
            if candidates:
                values = dict(entry_id=entry_id, namespace="default",
                              embedding_model=MODEL, cache_version="v1")
                assert writer.invalidate(**{selector: values[selector]}) == 1
            return result

        monkeypatch.setattr(semantic_cache, "find_most_similar", search)
        result = lookup(reader)
        assert not result.hit and result.reason == "invalidated"


def test_expiration_and_clear_span_versions_only_in_current_namespace(inputs):
    path, embedder = inputs
    now = [1000]
    with SemanticCache(path, embedder, clock=lambda: now[0]) as cache:
        save(cache, ttl_seconds=10)
        with SemanticCache(path, embedder, cache_version="v2", clock=lambda: now[0]) as new:
            save(new, ttl_seconds=20)
        with SemanticCache(path, embedder, namespace="other", clock=lambda: now[0]) as other:
            save(other, ttl_seconds=10)
        now[0] = 1010
        assert not lookup(cache).hit
        assert cache.remove_expired() == 1
        assert cache.clear() == 1
        assert len(inspect(path)[1]) == 1
        assert inspect(path)[1][0]["namespace"] == "other"


@pytest.mark.parametrize("version", ["", " ", None, True, 2])
def test_invalid_versions_fail_before_database_creation(inputs, version):
    path, embedder = inputs
    with pytest.raises(ValueError, match="version"):
        CacheConfig(cache_version=version)
    with pytest.raises(ValueError, match="version"):
        SemanticCache(path, embedder, cache_version=version)
    assert not path.exists()


def test_invalid_invalidation_selectors_do_not_delete(inputs):
    path, embedder = inputs
    with SemanticCache(path, embedder) as cache:
        save(cache)
        for selectors in ({}, {"entry_id": ""}, {"cache_version": 1}):
            with pytest.raises(ValueError):
                cache.invalidate(**selectors)
        assert lookup(cache).hit
