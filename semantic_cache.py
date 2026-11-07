"""SQLite semantic cache for explicitly approved, context-independent content.

The caller supplies an EmbeddingGenerator; this module never generates answers,
loads secrets, or imports the CLI. Eligibility is a trusted caller decision.
"""

from dataclasses import dataclass
import json
import math
from numbers import Real
from pathlib import Path
import sqlite3
import time
import uuid

import numpy as np

from cache_config import CacheConfig, DEFAULT_CONFIG, _UNSET, validate_threshold
from cache_schema import SCHEMA_VERSION, SCHEMA, create_expiration_index, migrate_v1
from embeddings import (
    DIMENSIONS, MODEL, TASK_TYPE, NORMALIZATION_VERSION, PREPROCESSING_VERSION,
)
from semantic_similarity import find_most_similar


VECTOR_DTYPE = "<f4"
ELIGIBILITY_VERSION = "explicit-review-v1"
EMBEDDING_CONFIG = json.dumps({
    "normalization": NORMALIZATION_VERSION,
    "preprocessing": PREPROCESSING_VERSION,
}, sort_keys=True, separators=(",", ":"))
SCOPE = """namespace = ? AND embedding_model = ? AND embedding_version = ?
AND embedding_task = ? AND embedding_dimensions = ? AND embedding_dtype = ?
AND eligibility_version = ? AND cache_version = ? AND embedding_config = ?"""


@dataclass(frozen=True, kw_only=True)
class QueryEligibility:
    """Explicit assessment bound to this exact query; all flags default to deny."""

    query: str
    context_independent: bool = False
    non_sensitive: bool = False
    time_independent: bool = False
    external_state_independent: bool = False

    def allows(self, query):
        return self.query == query and all(flag is True for flag in (
            self.context_independent, self.non_sensitive,
            self.time_independent, self.external_state_independent,
        ))


@dataclass(frozen=True, kw_only=True)
class ResponseEligibility:
    """Caller must assess API success, refusal status, and response privacy."""

    response: str
    successful: bool = False
    not_safety_refusal: bool = False
    non_sensitive: bool = False

    def allows(self, response):
        return self.response == response and all(flag is True for flag in (
            self.successful, self.not_safety_refusal, self.non_sensitive,
        ))


@dataclass(frozen=True)
class CacheLookup:
    hit: bool
    response: str | None = None
    matched_query: str | None = None
    similarity_score: float | None = None
    entry_id: str | None = None
    reason: str = "miss"
    skipped_corrupt_entries: int = 0


class CacheError(RuntimeError):
    """The cache database could not be used; callers may bypass the cache."""


class CacheConflictError(CacheError):
    """An unexpired exact query already has a different answer or model."""


def _nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def _eligible(query, eligibility):
    return (
        _nonempty(query) and isinstance(eligibility, QueryEligibility)
        and eligibility.allows(query)
    )


def _positive_seconds(value):
    if (
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
        or value <= 0 or not math.isfinite(value)
    ):
        raise ValueError("TTL must be a finite positive number of seconds.")
    return float(value)


def _validate_vector(value):
    """Apply the similarity module's normalization checks and fixed dimensions."""
    try:
        vector = np.asarray(value)
        if vector.shape != (DIMENSIONS,):
            raise ValueError
        find_most_similar(vector, {}, threshold=-1)
        return vector.astype(np.float64)
    except (ValueError, TypeError, OverflowError):
        raise ValueError("Expected a normalized, finite 768-dimensional vector.") from None


class CacheBehavior:
    """Eligibility, semantic matching and cache operations shared by both stores."""

    def _configure(self, embedder, *, threshold=_UNSET, config=DEFAULT_CONFIG, ttl_seconds=86400,
                   namespace="default", embedding_version="v1", cache_version=_UNSET, clock=time.time):
        if not isinstance(config, CacheConfig):
            raise ValueError("config must be a CacheConfig instance.")
        threshold = validate_threshold(
            config.similarity_threshold if threshold is _UNSET else threshold
        )
        if not _nonempty(namespace) or not _nonempty(embedding_version):
            raise ValueError("Namespace and embedding version must be nonempty.")
        cache_version = config.cache_version if cache_version is _UNSET else cache_version
        if not _nonempty(cache_version):
            raise ValueError("Cache version must be a nonempty string.")
        self.cache_version = cache_version
        self.threshold = threshold
        self.ttl_seconds = _positive_seconds(ttl_seconds)
        self.embedder = embedder
        self.clock = clock
        self._scope = (
            namespace, MODEL, embedding_version, TASK_TYPE, DIMENSIONS,
            VECTOR_DTYPE, ELIGIBILITY_VERSION, cache_version, EMBEDDING_CONFIG,
        )

    @property
    def namespace(self):
        return self._scope[0]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _db(self):
        if self._connection is None:
            raise CacheError("Cache is closed.")
        return self._connection

    def _now(self):
        now = self.clock()
        if isinstance(now, bool) or not isinstance(now, Real) or not math.isfinite(now):
            raise ValueError("Clock must return finite UTC epoch seconds.")
        return float(now)

    def _decode(self, row, now):
        """Corrupt rows are ignored, never deserialized as executable objects."""
        try:
            if not all(_nonempty(row[name]) for name in (
                "id", "query_key", "original_query", "cached_response", "response_model",
            )):
                return None
            if row["query_key"] != row["original_query"].strip():
                return None
            created, expires = row["created_at"], row["expires_at"]
            if not all(type(t) in (int, float) and math.isfinite(t) for t in (created, expires)):
                return None
            if not created <= now < expires:
                return None
            blob = row["embedding"]
            if not isinstance(blob, bytes) or len(blob) != DIMENSIONS * 4:
                return None
            return _validate_vector(np.frombuffer(blob, dtype=VECTOR_DTYPE))
        except (ValueError, TypeError, OverflowError):
            return None

    def lookup(self, query, *, eligibility=None, threshold=_UNSET):
        """Search approved entries; a threshold override affects only this lookup.

        Invalid thresholds raise ValueError before eligibility checks or API calls.
        skipped_corrupt_entries reports malformed eligible candidates without
        exposing their content. Storage failures still raise CacheError.
        """
        effective_threshold = validate_threshold(
            self.threshold if threshold is _UNSET else threshold
        )
        if not _eligible(query, eligibility):
            return CacheLookup(hit=False, reason="ineligible")
        self._db()
        # Uses the injected EmbeddingGenerator and its existing instrumentation.
        vector = _validate_vector(self.embedder.embed_query(query.strip()))
        now = self._now()  # Check expiration after embedding request latency.
        rows = self._candidates(now, query.strip())

        candidates, entries = {}, {}
        skipped_corrupt_entries = 0
        for row in rows:
            stored_vector = self._decode(row, now)
            if stored_vector is not None:
                candidates[row["original_query"]] = stored_vector
                entries[row["original_query"]] = row
            else:
                skipped_corrupt_entries += 1
        match = find_most_similar(vector, candidates, threshold=effective_threshold)
        if match is None:
            return CacheLookup(hit=False, skipped_corrupt_entries=skipped_corrupt_entries)
        row = entries[match.query]
        # Re-read after similarity search: another connection may have invalidated
        # or replaced this entry after the initial candidate snapshot.
        current = self._current(row["id"])
        if current is None or current != row:
            return CacheLookup(hit=False, reason="invalidated", skipped_corrupt_entries=skipped_corrupt_entries)
        # Do not return an entry that expired during a long local scan.
        if row["expires_at"] <= self._now():
            return CacheLookup(hit=False, reason="expired", skipped_corrupt_entries=skipped_corrupt_entries)
        return CacheLookup(
            hit=True, response=row["cached_response"], matched_query=match.query,
            similarity_score=match.score, entry_id=row["id"], reason="hit",
            skipped_corrupt_entries=skipped_corrupt_entries,
        )

    @staticmethod
    def _duplicate_id(row, response, model):
        if row["cached_response"] != response or row["response_model"] != model:
            raise CacheConflictError("An active query already has a different response or model.")
        return row["id"]

    def save(
        self, query, response, model, *, eligibility=None,
        response_eligibility=None, ttl_seconds=None,
    ):
        """Return entry ID, or None when ineligible. Active conflicts raise an error.

        Exact duplicates reuse their ID without refreshing TTL or calling the
        embedder. Expired/corrupt exact entries may be replaced after fresh review.
        """
        if not _eligible(query, eligibility) or not (
            _nonempty(response) and isinstance(response_eligibility, ResponseEligibility)
            and response_eligibility.allows(response)
        ):
            return None
        if not _nonempty(model):
            raise ValueError("The response model must be a nonempty string.")
        ttl = self.ttl_seconds if ttl_seconds is None else _positive_seconds(ttl_seconds)
        key = query.strip()
        existing = self._existing(key, self._now())
        if existing is not None:
            return self._duplicate_id(existing, response, model)

        vector = _validate_vector(self.embedder.embed_query(key))
        blob = vector.astype(VECTOR_DTYPE).tobytes()
        return self._store_blob(key, query, response, model, blob, ttl)

    def invalidate(
        self, *, entry_id=None, namespace=None, embedding_model=None,
        embedding_version=None, cache_version=None,
    ):
        """Delete matching rows in a namespace; filters combine with AND.

        Without an explicit namespace, use this instance's namespace. At least
        one selector is required; use clear() for an intentional current-scope wipe.
        This does not prevent a caller from saving a newly reviewed answer later.
        """
        selectors = {
            "id": entry_id, "namespace": namespace, "embedding_model": embedding_model,
            "embedding_version": embedding_version, "cache_version": cache_version,
        }
        if all(value is None for value in selectors.values()):
            raise ValueError("Invalidation requires at least one selector.")
        if any(value is not None and not _nonempty(value) for value in selectors.values()):
            raise ValueError("Invalidation selectors must be nonempty strings.")
        selectors["namespace"] = self.namespace if namespace is None else namespace
        selected = [(column, value) for column, value in selectors.items() if value is not None]
        where = " AND ".join(column + " = ?" for column, _ in selected)
        return self._delete(where, tuple(value for _, value in selected), "Cache invalidation failed.")

    def remove_expired(self):
        return self._delete("namespace = ? AND expires_at <= ?", (self.namespace, self._now()),
                            "Expired cache cleanup failed.")

    def clear(self):
        return self._delete("namespace = ?", (self.namespace,), "Cache clearing failed.")


class SemanticCache(CacheBehavior):
    """One connection per instance/thread; use as a context manager or close it."""

    def __init__(
        self, path, embedder, *, threshold=_UNSET, config=DEFAULT_CONFIG, ttl_seconds=86400,
        namespace="default", embedding_version="v1", cache_version=_UNSET, clock=time.time,
        connection_timeout=5, require_existing=False, allow_initialize=True,
    ):
        self._configure(embedder, threshold=threshold, config=config, ttl_seconds=ttl_seconds,
                        namespace=namespace, embedding_version=embedding_version,
                        cache_version=cache_version, clock=clock)
        if (type(connection_timeout) not in (int, float) or not math.isfinite(connection_timeout)
                or not 0 <= connection_timeout <= 60):
            raise ValueError("Connection timeout must be between 0 and 60 seconds.")
        if type(require_existing) is not bool or type(allow_initialize) is not bool:
            raise ValueError("Database opening policies must be boolean.")
        self._connection = None
        try:
            target = Path(path).absolute().as_uri()+"?mode=rw" if require_existing else path
            self._connection = sqlite3.connect(target, timeout=connection_timeout, uri=require_existing)
            self._connection.row_factory = sqlite3.Row
            with self._connection:
                self._connection.execute("BEGIN IMMEDIATE")
                version = self._connection.execute("PRAGMA user_version").fetchone()[0]
                if version == 0:
                    if not allow_initialize:
                        raise CacheError("Explicit database initialization is required.")
                    tables = self._connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table' "
                        "AND name NOT LIKE 'sqlite_%'"
                    ).fetchall()
                    if tables:
                        raise CacheError("Refusing to initialize an unrelated database.")
                    self._connection.execute(SCHEMA)
                    create_expiration_index(self._connection)
                    self._connection.execute("PRAGMA user_version = 2")
                elif version == 1:
                    migrate_v1(self._connection)
                elif version != SCHEMA_VERSION:
                    raise CacheError("Unsupported cache schema version.")
                # Ensure an existing database has the expected table/columns.
                self._connection.execute(
                    "SELECT id, query_key, original_query, cached_response, response_model, "
                    "embedding, created_at, expires_at FROM cache_entries WHERE " + SCOPE
                    + " LIMIT 0", self._scope,
                )
        except (sqlite3.Error, CacheError, ValueError):
            self.close()
            raise CacheError("Cache database initialization failed.") from None

    def _candidates(self, now, query_key):
        try:
            return self._db().execute(
                "SELECT * FROM cache_entries WHERE " + SCOPE
                + " AND expires_at > ? ORDER BY created_at, id",
                (*self._scope, now),
            ).fetchall()
        except sqlite3.Error:
            raise CacheError("Cache lookup failed.") from None

    def _current(self, entry_id):
        try:
            return self._db().execute(
                "SELECT * FROM cache_entries WHERE id = ? AND " + SCOPE,
                (entry_id, *self._scope),
            ).fetchone()
        except sqlite3.Error:
            raise CacheError("Cache lookup failed.") from None

    def _existing(self, query_key, now):
        try:
            row = self._db().execute(
                "SELECT * FROM cache_entries WHERE " + SCOPE + " AND query_key = ?",
                (*self._scope, query_key),
            ).fetchone()
        except sqlite3.Error:
            raise CacheError("Cache storage failed.") from None
        return row if row is not None and self._decode(row, now) is not None else None

    def _store_blob(self, key, query, response, model, blob, ttl):
        # Short write transaction: no API call occurs while holding a write lock.
        try:
            with self._db():
                self._db().execute("BEGIN IMMEDIATE")
                now = self._now()
                expires = now + ttl
                if not math.isfinite(expires) or expires <= now:
                    raise ValueError("TTL cannot be represented at this timestamp.")
                existing = self._existing(key, now)  # Recheck after concurrent writers.
                if existing is not None:
                    return self._duplicate_id(existing, response, model)
                self._db().execute(
                    "DELETE FROM cache_entries WHERE " + SCOPE + " AND query_key = ?",
                    (*self._scope, key),
                )
                entry_id = uuid.uuid4().hex
                self._db().execute(
                    """INSERT INTO cache_entries (
                        id, namespace, query_key, original_query, cached_response,
                        response_model, embedding, embedding_model, embedding_version,
                        embedding_task, embedding_dimensions, embedding_dtype,
                        eligibility_version, cache_version, embedding_config, created_at, expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (entry_id, self.namespace, key, query, response, model, blob,
                     *self._scope[1:], now, expires),
                )
            return entry_id
        except sqlite3.Error:
            raise CacheError("Cache storage failed.") from None

    def _delete(self, where, params, message):
        try:
            with self._db():
                cursor = self._db().execute("DELETE FROM cache_entries WHERE " + where, params)
            return cursor.rowcount
        except sqlite3.Error:
            raise CacheError(message) from None
