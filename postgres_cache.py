"""Private PostgreSQL cache storage using transaction-local settings."""
from contextlib import closing, contextmanager
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import uuid

import psycopg
from psycopg.rows import dict_row

from database import DatabaseError
from postgres_catalog import SCOPE_LIMIT, verify_catalog
from semantic_cache import CacheBehavior, CacheConflictError, CacheError, SCOPE

TABLE = "lcr_cache.cache_entries"
PG_SCOPE = SCOPE.replace("?", "%s")
INSERT = f"""INSERT INTO {TABLE} (
    id, namespace, query_key, query_digest, original_query, cached_response, response_model,
    embedding, embedding_model, embedding_version, embedding_task, embedding_dimensions,
    embedding_dtype, eligibility_version, cache_version, embedding_config, created_at, expires_at
) VALUES ({", ".join(["%s"] * 18)})
ON CONFLICT ON CONSTRAINT cache_scope_digest_unique DO NOTHING RETURNING id"""
MIGRATION = Path(__file__).resolve().parent / "migrations/postgresql/001_cache.sql"
DIGEST_MIGRATION = MIGRATION.with_name("002_digest.sql")


def _digest(key):
    return hashlib.sha256(key.encode("utf-8")).digest()


def _database_now(cursor):
    cursor.execute("SELECT EXTRACT(EPOCH FROM clock_timestamp())::double precision AS now")
    return cursor.fetchone()["now"]


def _connect(config):
    try:
        return psycopg.connect(config.dsn, sslmode="verify-full", gssencmode="disable",
                               connect_timeout=5, prepare_threshold=None, autocommit=True,
                               row_factory=dict_row)
    except (psycopg.Error, ValueError):
        raise DatabaseError("PostgreSQL connection failed.") from None


@contextmanager
def _transaction(connection):
    """The transaction pooler can reuse its server connection immediately after exit."""
    try:
        with connection.transaction(), connection.cursor() as cursor:
            cursor.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
            cursor.execute("SET LOCAL statement_timeout = '5s'")
            cursor.execute("SET LOCAL lock_timeout = '2s'")
            yield cursor
    except psycopg.Error:
        raise CacheError("PostgreSQL cache transaction failed.") from None


def _version(cursor):
    cursor.execute("SELECT version FROM lcr_cache.schema_version")
    rows = cursor.fetchall()
    if len(rows) != 1 or rows[0]["version"] not in (1, 2):
        raise DatabaseError("Unsupported PostgreSQL cache migration version.")
    return rows[0]["version"]


def _verify(cursor, expected=2):
    if _version(cursor) != expected:
        raise DatabaseError("PostgreSQL cache migration required.")
    verify_catalog(cursor, expected)


def migrate(config):
    """Explicit, atomic installation or upgrade. Refuse to adopt an unfamiliar schema."""
    try:
        with closing(_connect(config)) as connection, _transaction(connection) as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(1684027551)")
            cursor.execute("SELECT to_regnamespace('lcr_cache') IS NOT NULL AS present")
            if not cursor.fetchone()["present"]:
                cursor.execute(MIGRATION.read_text(encoding="utf-8"), prepare=False)
            if _version(cursor) == 1:
                _verify(cursor, 1)
                cursor.execute(DIGEST_MIGRATION.read_text(encoding="utf-8"), prepare=False)
            _verify(cursor)
    except (OSError, psycopg.Error, CacheError):
        raise DatabaseError("PostgreSQL cache migration failed.") from None


def verify(config):
    try:
        with closing(_connect(config)) as connection, _transaction(connection) as cursor:
            _verify(cursor)
    except (psycopg.Error, CacheError):
        raise DatabaseError("PostgreSQL cache verification failed.") from None
    return dict(backend="PostgreSQL", connection="Verified", schema_version=2,
                checked_at_utc=datetime.now(timezone.utc).isoformat())


class PostgresCache(CacheBehavior):
    def __init__(self, database_config, embedder, **options):
        self._configure(embedder, **options)
        if any(len(value.encode("utf-8")) > SCOPE_LIMIT for value in self._scope if isinstance(value, str)):
            raise ValueError("PostgreSQL text scope fields must fit in 128 UTF-8 bytes.")
        self._connection = _connect(database_config)
        try:
            with _transaction(self._connection) as cursor:
                _verify(cursor)
        except (CacheError, DatabaseError):
            self.close()
            raise DatabaseError("PostgreSQL cache schema could not be verified.") from None

    def close(self):
        try:
            super().close()
        except psycopg.Error:
            raise CacheError("PostgreSQL cache close failed.") from None

    def _now(self):
        with _transaction(self._db()) as cursor:
            return _database_now(cursor)

    def _candidates(self, now, query_key):
        with _transaction(self._db()) as cursor:
            cursor.execute(f"SELECT * FROM {TABLE} WHERE {PG_SCOPE} "
                           "AND expires_at > %s ORDER BY created_at, id", (*self._scope, now))
            rows = cursor.fetchall()
        digest = _digest(query_key)
        if any(row["query_digest"] == digest and row["query_key"] != query_key for row in rows):
            raise CacheConflictError("Cache query digest collision.")
        return rows

    def _current(self, entry_id):
        with _transaction(self._db()) as cursor:
            cursor.execute(f"SELECT * FROM {TABLE} WHERE id = %s AND {PG_SCOPE}", (entry_id, *self._scope))
            return cursor.fetchone()

    def _exact(self, cursor, key, *, lock=False):
        suffix = " FOR UPDATE" if lock else ""
        cursor.execute(f"SELECT * FROM {TABLE} WHERE {PG_SCOPE} AND query_digest = %s" + suffix,
                       (*self._scope, _digest(key)))
        row = cursor.fetchone()
        if row is not None and row["query_key"] != key:
            raise CacheConflictError("Cache query digest collision.")
        return row

    def _existing(self, query_key, now):
        with _transaction(self._db()) as cursor:
            row = self._exact(cursor, query_key)
        return row if row is not None and self._decode(row, now) is not None else None

    def _store_blob(self, key, query, response, model, blob, ttl):
        with _transaction(self._db()) as cursor:
            # A racing insert may commit between our pre-check and insert. Let
            # the unique constraint arbitrate, then inspect the committed winner.
            for _ in range(2):
                now = _database_now(cursor)
                expires = now + ttl
                if not math.isfinite(expires) or expires <= now:
                    raise ValueError("TTL cannot be represented at this timestamp.")
                entry_id = uuid.uuid4().hex
                cursor.execute(INSERT, (entry_id, self.namespace, key, _digest(key), query, response, model,
                                        blob, *self._scope[1:], now, expires))
                if cursor.fetchone() is not None:
                    return entry_id
                row = self._exact(cursor, key, lock=True)
                if row is not None:
                    now = _database_now(cursor)
                    if row["created_at"] > now and row["expires_at"] > now:
                        raise CacheConflictError("Active cache entry has a future creation time.")
                    if self._decode(row, now) is not None:
                        return self._duplicate_id(row, response, model)
                    cursor.execute(f"DELETE FROM {TABLE} WHERE id = %s AND {PG_SCOPE}",
                                   (row["id"], *self._scope))
            raise CacheError("Concurrent cache changes prevented storage.")

    def _delete(self, where, params, message):
        # where is constructed only from CacheBehavior's fixed selector names.
        with _transaction(self._db()) as cursor:
            cursor.execute(f"DELETE FROM {TABLE} WHERE " + where.replace("?", "%s"), params)
            return cursor.rowcount
