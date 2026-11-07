"""Opt-in test for an EMPTY disposable PostgreSQL database on loopback with TLS."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import os
from threading import Barrier
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
from database import initialize_database, open_cache, verify_database
from database_config import PostgresConfig
import postgres_cache as pg
from semantic_cache import CacheConflictError
from test_semantic_cache import lookup, save, vector


def test_disposable_local_postgres(monkeypatch):
    if os.environ.get("LCR_TEST_LOCAL_PG") != "1" or not os.environ.get("LCR_TEST_LOCAL_PG_DSN"):
        pytest.skip("Set the explicit local PostgreSQL test flag and disposable database DSN.")
    config = PostgresConfig(os.environ["LCR_TEST_LOCAL_PG_DSN"])
    url = urlsplit(config.dsn)
    if url.hostname != "127.0.0.1" or url.path != "/lcr_adapter_test":
        pytest.fail("Test requires a disposable lcr_adapter_test database on IPv4 loopback.")

    # Pin the network destination even when libpq has ambient hostaddr settings.
    connect = pg.psycopg.connect
    monkeypatch.setattr(pg.psycopg, "connect",
                        lambda *args, **kwargs: connect(*args, hostaddr="127.0.0.1", **kwargs))
    with closing(pg._connect(config)) as conn, conn.cursor() as cursor:
        cursor.execute("SELECT to_regnamespace('lcr_cache') IS NOT NULL AS present")
        assert not cursor.fetchone()["present"], "Use a fresh disposable database."
    initialize_database(config)
    assert verify_database(config)["connection"] == "Verified"
    gate = Barrier(2, timeout=10)
    def embed(_):
        gate.wait()
        return vector()
    def write(answer):
        with open_cache(SimpleNamespace(embed_query=embed), database_config=config) as cache:
            try:
                return save(cache, answer=answer)
            except CacheConflictError:
                return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(write, ["First fixture", "Second fixture"]))
    assert results.count(None) == 1
    with open_cache(SimpleNamespace(embed_query=lambda _:vector()), database_config=config) as cache:
        winner = next(i for i, entry_id in enumerate(results) if entry_id is not None)
        result = lookup(cache)
        assert result.hit and result.entry_id == results[winner]
        assert result.response == ["First fixture", "Second fixture"][winner]
        assert cache.clear() == 1
