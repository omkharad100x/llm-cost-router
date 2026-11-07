"""Offline Psycopg boundary tests; PostgreSQL catalog and locks are simulated."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import random
import re
import socket
import sqlite3
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import postgres_cache as pg
from database import DatabaseError, initialize_database, open_cache, verify_database
from database_config import DatabaseConfig, PostgresConfig, database_summary, load_database_config
from semantic_cache import CacheConflictError, CacheError
from test_semantic_cache import save, lookup, vector

DSN = "postgresql://worker:fixture-only@pool.invalid:6543/fixture"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Network forbidden")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(pg.psycopg, "connect", denied)


class DriverDouble:
    def __init__(self, path, log, server_clock):
        self.db = sqlite3.connect(path, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.create_collation("C", lambda a, b: (a > b) - (a < b))
        self.db.create_function("octet_length", 1, lambda v: len(v.encode("utf-8")))
        self.db.create_function("convert_to", 2, lambda v, enc: v.encode("utf-8"))
        self.db.create_function("sha256", 1, lambda v: hashlib.sha256(v).digest())
        self.server_clock = server_clock
        self.log = log
        self.closed = False

    @contextmanager
    def transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    @contextmanager
    def cursor(self):
        cursor = CursorDouble(self)
        try:
            yield cursor
        finally:
            cursor.sql.close()

    def close(self):
        self.db.close()
        self.closed = True


class CursorDouble:
    def __init__(self, driver):
        self.driver = driver
        self.sql = driver.db.cursor()
        self.catalog = None

    def catalog_rows(self, table, sql):
        db = self.driver.db
        definition = db.execute("SELECT sql FROM sqlite_master WHERE name=?", (table,)).fetchone()[0]
        if "pg_catalog.pg_index" in sql:
            rows = []
            # SQLite's INTEGER PRIMARY KEY uses the rowid, not a separate index.
            if table == "schema_version":
                rows.append(dict(kind="p", columns=["version"], is_unique=True, usable=True, name="meta_pk"))
            for idx in db.execute("SELECT * FROM pragma_index_list(?)", (table,)):
                cols = [r["name"] for r in db.execute(
                    "SELECT name FROM pragma_index_info(?) ORDER BY seqno", (idx["name"],))]
                kind = {"pk":"p", "u":"u", "c":None}[idx["origin"]]
                rows.append(dict(is_unique=bool(idx["unique"]), kind=kind, columns=cols,
                                 usable=True, name="cache_scope_digest_unique" if kind=="u" else None))
            return rows
        if "pg_catalog.pg_attribute" in sql:
            return [dict(name=r["name"], type=r["type"].lower(), plain=True, required=bool(r["notnull"] or r["pk"]),
                         c_collation=f'{r["name"]} TEXT COLLATE "C"' in definition)
                    for r in db.execute("SELECT * FROM pragma_table_info(?)", (table,))]
        return [dict(expression=e.replace("trim(", "btrim("), valid=True)
                for e in re.findall(r"CHECK \((.*)\)(?:,|$)", definition, re.MULTILINE)]

    def upgrade(self, migration):
        # Emulate ALTER CONSTRAINT using a transactional SQLite table rebuild.
        db = self.driver.db
        ddl = db.execute("SELECT sql FROM sqlite_master WHERE name='cache_entries'").fetchone()[0]
        db.execute("ALTER TABLE cache_entries RENAME TO prior_entries")
        db.execute("DROP INDEX cache_expiration")
        ddl = ddl.replace("original_query TEXT", "query_digest BYTEA NOT NULL, original_query TEXT")
        checks = re.findall(r"ADD CONSTRAINT \w+ (CHECK \([^\n]+\))", migration.partition("-- PostgreSQL generated")[0])
        db.execute(ddl.replace("UNIQUE (namespace, query_key,", ",\n".join(checks) + ",\n UNIQUE (namespace, query_digest,"))
        names = ", ".join(r["name"] for r in db.execute("PRAGMA table_info(prior_entries)"))
        db.execute(f"INSERT INTO cache_entries ({names},query_digest) SELECT {names}, "
                   "sha256(convert_to(query_key,'UTF8')) FROM prior_entries")
        db.execute("DROP TABLE prior_entries")
        db.execute("CREATE INDEX cache_expiration ON cache_entries(namespace, expires_at)")
        db.execute("DROP TABLE schema_version")
        db.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY CHECK (version = 2)\n)")
        db.execute("INSERT INTO schema_version VALUES (2)")

    def execute(self, sql, params=None, **kwargs):
        self.driver.log.append(sql)
        self.catalog = None
        if sql.startswith(("SET ", "SELECT pg_advisory_xact_lock")):
            return
        if "to_regnamespace" in sql:
            sql = "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE name='schema_version') AS present"
        if sql.startswith("SELECT current_setting"):
            sql = "SELECT 'UTF8' AS encoding"
        if sql.startswith("SELECT EXTRACT"):
            sql, params = "SELECT ? AS now", (self.driver.server_clock[0],)
        try:
            if sql.startswith("-- The runner verifies"):
                self.upgrade(sql)
                return
            if params and "pg_catalog." in sql:
                self.catalog = self.catalog_rows(params[0].split(".")[-1], sql)
                return
            for statement in sql.split(";"):
                statement = statement.strip()
                if not statement or statement.startswith(("CREATE SCHEMA", "REVOKE")):
                    continue
                statement = statement.replace("ON CONFLICT ON CONSTRAINT cache_scope_digest_unique",
                    "ON CONFLICT (namespace,query_digest,embedding_model,embedding_version,embedding_task,"
                    "embedding_dimensions,embedding_dtype,eligibility_version,cache_version,embedding_config)")
                statement = statement.replace("lcr_cache.", "").replace("%s", "?").replace(" FOR UPDATE", "")
                self.sql.execute(statement, params or ())
        except sqlite3.Error:
            raise pg.psycopg.DatabaseError("synthetic driver failure") from None

    @property
    def rowcount(self):
        return self.sql.rowcount

    def fetchone(self):
        row = self.sql.fetchone()
        return dict(row) if row is not None else None

    def fetchall(self):
        return self.catalog if self.catalog is not None else [dict(row) for row in self.sql.fetchall()]


@pytest.fixture
def backend(tmp_path, monkeypatch):
    path = tmp_path / "standin.sqlite3"
    calls, log, connections, server_clock = [], [], [], [1000.0]
    def connect(dsn, **kwargs):
        calls.append(kwargs)
        connection = DriverDouble(path, log, server_clock)
        connections.append(connection)
        return connection
    monkeypatch.setattr(pg.psycopg, "connect", connect)
    return SimpleNamespace(config=load_database_config({"LCR_DATABASE_BACKEND":"postgresql",
                           "LCR_POSTGRES_DSN":DSN, "LCR_DATABASE_MODE":"production"}),
                           server_clock=server_clock, path=path, calls=calls, log=log, connections=connections)


@pytest.mark.parametrize("dsn", ["", "file.sqlite", "postgresql://host/db", DSN+"?sslmode=prefer",
                                 DSN+"?options=-c", DSN+"?hostaddr=127.0.0.1",
                                 DSN+"?sslmode=verify-full&sslmode=disable"])
def test_invalid_postgres_configuration_is_sanitized(dsn):
    with pytest.raises(ValueError, match="Invalid PostgreSQL") as error:
        load_database_config({"LCR_DATABASE_BACKEND":"postgresql", "LCR_POSTGRES_DSN":dsn})
    assert "fixture-only" not in str(error.value)


def test_selection_does_not_fall_back():
    assert isinstance(load_database_config({}), DatabaseConfig)
    with pytest.raises(ValueError):
        load_database_config({"LCR_DATABASE_BACKEND":"postgres"})
    with pytest.raises(ValueError):
        load_database_config({"LCR_DATABASE_BACKEND":"postgresql", "LCR_DATABASE_PATH":"ignored.sqlite3"})
    env = {"LCR_DATABASE_BACKEND":"postgresql", "LCR_POSTGRES_DSN":DSN}
    assert database_summary(env)["backend"] == "PostgreSQL"
    assert "fixture-only" not in repr(load_database_config(env))


def test_transport_settings_and_idempotent_migration(backend):
    with pytest.raises(DatabaseError):
        verify_database(backend.config)
    initialize_database(backend.config)
    initialize_database(backend.config)
    assert verify_database(backend.config)["schema_version"] == 2
    assert sum("CREATE SCHEMA" in sql for sql in backend.log) == 1
    assert all(c.closed for c in backend.connections)
    assert all(k["prepare_threshold"] is None and k["sslmode"] == "verify-full" and
               k["autocommit"] is True and k["connect_timeout"] == 5 for k in backend.calls)
    assert "SET LOCAL lock_timeout = '2s'" in backend.log
    assert "SET LOCAL statement_timeout = '5s'" in backend.log


def test_connection_failure_is_sanitized(monkeypatch):
    def fail(*args, **kwargs):
        raise pg.psycopg.OperationalError(DSN)
    monkeypatch.setattr(pg.psycopg, "connect", fail)
    with pytest.raises(DatabaseError, match="connection failed") as error:
        open_cache(None, database_config=PostgresConfig(DSN))
    assert "fixture-only" not in str(error.value) and error.value.__suppress_context__


def test_failed_migration_rolls_back_and_retries(backend, monkeypatch):
    with monkeypatch.context() as patch:
        def fail(cursor, expected=2):
            raise pg.psycopg.OperationalError(DSN)
        patch.setattr(pg, "_verify", fail)
        with pytest.raises(DatabaseError, match="migration failed"):
            initialize_database(backend.config)
    assert backend.connections[-1].closed
    initialize_database(backend.config)
    with open_cache(None, database_config=backend.config):
        pass


@pytest.mark.parametrize("options", [{"namespace":"other"}, {"namespace":"DEFAULT"}, {"embedding_version":"next"}, {"cache_version":"next"}])
def test_scope_conflict_expiry_and_reopening(backend, options):
    initialize_database(backend.config)
    embed = SimpleNamespace(embed_query=Mock(return_value=vector()))
    def opened(**kwargs):
        return open_cache(embed, database_config=backend.config, **kwargs)
    with opened() as a, opened(**options) as b:
        entry = save(a, ttl_seconds=1)
        assert save(a) == entry
        assert embed.embed_query.call_count == 1
        with pytest.raises(CacheConflictError):
            save(a, answer="Conflicting answer")
        assert not lookup(b).hit
        second = save(b, answer="Other scope")
        assert lookup(b).entry_id == second and lookup(a).entry_id == entry
        backend.server_clock[0] += 1
        assert not lookup(a).hit
        assert a.remove_expired() == 1
    with opened(**options) as b:
        assert lookup(b).entry_id == second
        assert b.invalidate(entry_id=second) == 1 and not lookup(b).hit


def test_corrupt_row_skip_repair_and_final_recheck(backend, monkeypatch):
    import semantic_cache
    initialize_database(backend.config)
    emb = SimpleNamespace(embed_query=lambda _:vector())
    with open_cache(emb, database_config=backend.config) as cache:
        bad = save(cache)
        cache._db().db.execute("UPDATE cache_entries SET embedding=? WHERE id=?", (b"bad", bad))
        result = lookup(cache)
        assert not result.hit and result.skipped_corrupt_entries == 1
        repaired = save(cache)
        assert repaired != bad
        score = semantic_cache.find_most_similar
        def invalidate_after_scoring(vector, candidates, **kwargs):
            match = score(vector, candidates, **kwargs)
            if match:
                assert cache.clear() == 1
            return match
        monkeypatch.setattr(semantic_cache, "find_most_similar", invalidate_after_scoring)
        assert lookup(cache).reason == "invalidated"


@pytest.mark.parametrize("conflict", [False, True])
def test_two_writers_recheck_winning_insert(backend, conflict):
    initialize_database(backend.config)
    gate = Barrier(2, timeout=10)
    def embed(_):
        gate.wait()
        return vector()
    def writer(answer):
        with open_cache(SimpleNamespace(embed_query=embed), database_config=backend.config) as cache:
            try:
                return save(cache, answer=answer)
            except CacheConflictError:
                return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        answers = ["First", "Second" if conflict else "First"]
        ids = list(pool.map(writer, answers))
    if conflict:
        assert ids.count(None) == 1
    else:
        assert ids[0] is not None and ids[0] == ids[1]
    with open_cache(SimpleNamespace(embed_query=lambda _:vector()), database_config=backend.config) as cache:
        row = lookup(cache)
        assert row.hit and row.entry_id in [entry for entry in ids if entry is not None]
        assert row.response == answers[ids.index(row.entry_id)]
        assert cache.clear() == 1


def test_failed_expired_replacement_rolls_back(backend):
    initialize_database(backend.config)
    with open_cache(SimpleNamespace(embed_query=lambda _:vector()), database_config=backend.config) as cache:
        first = save(cache, ttl_seconds=1)
        backend.server_clock[0] += 2
        db = cache._db().db
        db.execute("CREATE TRIGGER reject BEFORE INSERT ON cache_entries WHEN NOT EXISTS "
                   "(SELECT 1 FROM cache_entries) BEGIN SELECT RAISE(ABORT,'fixture'); END")
        with pytest.raises(CacheError, match="transaction failed"):
            save(cache)
        assert db.execute("SELECT id FROM cache_entries").fetchone()[0] == first
        db.execute("DROP TRIGGER reject")
        assert save(cache) != first


def test_schema_drift_refused(backend):
    initialize_database(backend.config)
    with sqlite3.connect(backend.path) as db:
        db.execute("DROP INDEX cache_expiration")
    with pytest.raises(DatabaseError, match="indexes"):
        initialize_database(backend.config)
    with pytest.raises(DatabaseError):
        open_cache(None, database_config=backend.config)
    assert backend.connections[-1].closed


def test_incompatible_embedding_metadata_excluded(backend):
    initialize_database(backend.config)
    emb = SimpleNamespace(embed_query=lambda _:vector())
    with open_cache(emb, database_config=backend.config) as cache:
        entry = save(cache)
        cache._db().db.execute("UPDATE cache_entries SET embedding_model=? WHERE id=?", ("different", entry))
        assert not lookup(cache).hit
        assert cache.invalidate(embedding_model="different") == 1


def test_long_incompressible_keys(backend):
    initialize_database(backend.config)
    prefix = random.Random(8).randbytes(32768).hex() + "猫"
    with open_cache(SimpleNamespace(embed_query=lambda _:vector()), database_config=backend.config) as cache:
        for suffix in ("𐍈", "界"):
            key = prefix + suffix
            entry = save(cache, query=" "+key+" ", answer=suffix)
            row = cache._db().db.execute("SELECT * FROM cache_entries WHERE id=?", (entry,)).fetchone()
            assert row["query_key"] == key and row["original_query"] == " "+key+" "
            assert row["query_digest"] == hashlib.sha256(key.encode()).digest()
            assert save(cache, query=key, answer=suffix) == entry


@pytest.mark.parametrize("field", ["namespace", "cache_version", "embedding_version"])
def test_utf8_byte_budget(backend, field):
    initialize_database(backend.config)
    limit = "é"*64
    with open_cache(SimpleNamespace(embed_query=lambda _:vector()), database_config=backend.config,
                    **{field:limit}) as cache:
        entry = save(cache)
        with pytest.raises(sqlite3.IntegrityError):
            cache._db().db.execute(f"UPDATE cache_entries SET {field}=? WHERE id=?", (limit+"x", entry))
    before = len(backend.connections)
    with pytest.raises(ValueError, match="128 UTF-8 bytes"):
        open_cache(None, database_config=backend.config, **{field:limit+"x"})
    assert len(backend.connections) == before


def test_forced_collision(backend, monkeypatch):
    initialize_database(backend.config)
    with open_cache(SimpleNamespace(embed_query=lambda _:vector()), database_config=backend.config) as cache:
        entry = save(cache, query="first", answer="first answer", ttl_seconds=1)
        digest = pg._digest("first")
        monkeypatch.setattr(pg, "_digest", lambda key:digest)
        cache._db().db.create_function("sha256", 1, lambda _:digest)
        with pytest.raises(CacheConflictError, match="digest collision"):
            lookup(cache, "second")
        with pytest.raises(CacheConflictError, match="digest collision"):
            save(cache, query="second", answer="first answer")
        backend.server_clock[0] += 2
        # Force the insert-time recheck, including an expired conflicting row.
        monkeypatch.setattr(cache, "_existing", lambda key, now:None)
        with pytest.raises(CacheConflictError, match="digest collision"):
            save(cache, query="second", answer="different")
        row = cache._db().db.execute("SELECT id,query_key,cached_response FROM cache_entries").fetchone()
        assert tuple(row) == (entry, "first", "first answer")


def test_skewed_writers_and_future_creation(backend):
    initialize_database(backend.config)
    embed = SimpleNamespace(embed_query=lambda _:vector())
    with open_cache(embed, database_config=backend.config, clock=lambda:1001) as a, open_cache(
            embed, database_config=backend.config, clock=lambda:900) as b:
        entry = save(a, answer="winner", ttl_seconds=100)
        row = a._db().db.execute("SELECT created_at,expires_at FROM cache_entries").fetchone()
        assert tuple(row) == (1000, 1100)
        assert lookup(b).entry_id == entry
        with pytest.raises(CacheConflictError):
            save(b, answer="challenger")
        backend.server_clock[0] = 999
        with pytest.raises(CacheConflictError, match="future creation"):
            save(b, answer="challenger")
        backend.server_clock[0] = 1001
        assert lookup(b).response == "winner" and lookup(b).entry_id == entry


def test_namespace_is_immutable(backend):
    initialize_database(backend.config)
    emb = SimpleNamespace(embed_query=lambda _:vector())
    with open_cache(emb, database_config=backend.config, namespace="first") as a, open_cache(
            emb, database_config=backend.config, namespace="second") as b:
        with pytest.raises(AttributeError):
            a.namespace = "second"
        entry, other = save(a), save(b, answer="Second scope")
        assert lookup(a).entry_id == entry
        assert a.invalidate(entry_id=other) == 0
        assert a.clear() == 1 and lookup(b).entry_id == other


@pytest.mark.parametrize("defect", ["type", "nullable", "collation", "check", "unvalidated",
                                     "deferrable", "wrong_arbiter", "wrong_key"])
def test_catalog_drift_fails_closed(backend, monkeypatch, defect):
    initialize_database(backend.config)
    original = CursorDouble.fetchall
    def drift(cursor):
        rows = original(cursor)
        for row in rows:
            if row.get("name") == "query_key":
                if defect == "type": row["type"] = "character varying"
                if defect == "nullable": row["required"] = False
                if defect == "collation": row["c_collation"] = False
            if "expression" in row:
                if defect == "check": row["expression"] = "true"
                if defect == "unvalidated": row["valid"] = False
            if row.get("kind") == "u":
                if defect == "deferrable": row["usable"] = False
                if defect == "wrong_arbiter": row["name"] = "other"
                if defect == "wrong_key": row["columns"][1] = "query_key"
        return rows
    monkeypatch.setattr(CursorDouble, "fetchall", drift)
    with pytest.raises(DatabaseError):
        verify_database(backend.config)
    assert backend.connections[-1].closed


@pytest.mark.parametrize("oversized", [False, True])
def test_legacy_upgrade_is_atomic(backend, oversized):
    with pg.closing(pg._connect(backend.config)) as conn, pg._transaction(conn) as cursor:
        cursor.execute(pg.MIGRATION.read_text())
        namespace = "x"*129 if oversized else "old"
        values = ("legacy", namespace, "old query", "old query", "answer", "model", b"bytes",
                  "embed", "v1", "task", 768, "<f4", "review", "v1", "{}", 10, 20)
        cursor.execute("INSERT INTO lcr_cache.cache_entries VALUES (" + ",".join(["%s"]*17)+")", values)
    if oversized:
        with pytest.raises(DatabaseError):
            initialize_database(backend.config)
    else:
        initialize_database(backend.config)
    with sqlite3.connect(backend.path) as db:
        assert db.execute("SELECT version FROM schema_version").fetchone()[0] == (1 if oversized else 2)
        if oversized:
            assert db.execute("SELECT * FROM cache_entries").fetchone() == values
        else:
            assert db.execute("SELECT query_digest,embedding,id FROM cache_entries").fetchone() == (
                hashlib.sha256(b"old query").digest(), b"bytes", "legacy")
