"""Offline production-mode SQLite configuration, lifecycle and migration tests."""
from contextlib import closing
from dataclasses import replace
import json
from pathlib import Path
import socket
import sqlite3
from unittest.mock import Mock
import pytest

import database
import cache_schema
from database import DatabaseError, initialize_database, open_cache, verify_database
from database_config import DatabaseConfig, load_database_config, database_summary
from test_semantic_cache import vector, save, lookup
from test_cache_versioning import legacy_database, inspect

@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No network calls.")
    monkeypatch.setattr(socket,"create_connection",forbidden)
    monkeypatch.setattr(socket.socket,"connect",forbidden)
    original=Path.open
    def safe(path,*args,**kwargs):
        assert path.name not in (".env","memory.txt","secrets.toml")
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",safe)

@pytest.fixture
def config(tmp_path):
    return DatabaseConfig(mode="production",path=tmp_path/"cache.sqlite3",timeout_seconds=0)

@pytest.fixture
def embedder():
    obj=Mock(); obj.embed_query.return_value=vector()
    return obj

def test_defaults_and_environment_allowlist():
    class Source(dict):
        def get(self,key,default=None):
            assert key in ("LCR_DATABASE_MODE","LCR_DATABASE_PATH","LCR_DATABASE_TIMEOUT_SECONDS")
            return super().get(key,default)
    config=load_database_config(Source())
    assert config.mode=="development" and config.path.is_absolute()
    assert config.timeout_seconds==5

@pytest.mark.parametrize("changes",[
    {"LCR_DATABASE_MODE":"production"}, {"LCR_DATABASE_MODE":"unknown"},
    {"LCR_DATABASE_PATH":""}, {"LCR_DATABASE_PATH":"relative.sqlite3"},
    {"LCR_DATABASE_PATH":"postgresql://fake:synthetic@host/db"},
    {"LCR_DATABASE_TIMEOUT_SECONDS":"nan"}, {"LCR_DATABASE_TIMEOUT_SECONDS":"inf"},
    {"LCR_DATABASE_TIMEOUT_SECONDS":"-1"}, {"LCR_DATABASE_TIMEOUT_SECONDS":"61"},
    {"LCR_DATABASE_TIMEOUT_SECONDS":"synthetic-sensitive-value"},
])
def test_bad_configuration_is_explicit_and_safe(changes):
    with pytest.raises(ValueError) as caught: load_database_config(changes)
    assert "synthetic" not in str(caught.value)
    assert database_summary(changes)["configuration"]=="Invalid"

@pytest.mark.parametrize("value",[True,-1,float("inf"),float("nan"),61,"5"])
def test_direct_timeout_validation(config,value):
    with pytest.raises(ValueError): replace(config,timeout_seconds=value)

def test_production_cannot_create_missing_or_empty_database(config,embedder):
    with pytest.raises(DatabaseError): open_cache(embedder,database_config=config)
    assert not config.path.exists()
    config.path.touch()
    with pytest.raises(DatabaseError): open_cache(embedder,database_config=config)
    assert config.path.stat().st_size==0
    embedder.embed_query.assert_not_called()

def test_explicit_initialization_reopen_and_namespace_isolation(config,embedder):
    initialize_database(config)
    with open_cache(embedder,database_config=config,namespace="app-a") as a:
        entry=save(a)
        assert a._db().execute("PRAGMA busy_timeout").fetchone()[0]==0
        assert a._db().execute("PRAGMA synchronous").fetchone()[0]==2
        with open_cache(embedder,database_config=config,namespace="app-b") as b:
            assert not lookup(b).hit
            save(b)
            assert b.clear()==1
            assert lookup(a).entry_id==entry
    with pytest.raises(Exception): a._db()
    with open_cache(embedder,database_config=config,namespace="app-a") as reopened:
        assert lookup(reopened).entry_id==entry
    assert verify_database(config)["connection"]=="Verified"

def test_init_refuses_existing_database_and_preserves_rows(config,embedder):
    initialize_database(config)
    with open_cache(embedder,database_config=config) as cache: save(cache)
    before=inspect(config.path)
    with pytest.raises(DatabaseError): initialize_database(config)
    assert inspect(config.path)==before

def test_development_creation_preserves_old_workflow(config,embedder):
    dev=replace(config,mode="development")
    with open_cache(embedder,database_config=dev) as cache: save(cache)
    assert config.path.exists()

def test_verify_is_read_only_and_never_initializes(config):
    with pytest.raises(DatabaseError): verify_database(config)
    assert not config.path.exists()
    config.path.touch()
    with pytest.raises(DatabaseError): verify_database(config)
    assert config.path.stat().st_size==0

def test_probe_does_not_migrate_but_open_migrates_transactionally(config,embedder):
    legacy_database(config.path,embedder)
    before=inspect(config.path)
    with pytest.raises(DatabaseError): verify_database(config)
    assert inspect(config.path)==before
    with open_cache(embedder,database_config=config,clock=lambda:1000) as cache:
        assert lookup(cache).entry_id=="live"
    assert verify_database(config)["schema_version"]==2
    for old,new in zip(before[1],inspect(config.path)[1]):
        assert {key:new[key] for key in old}==old

def test_migration_failure_can_be_retried(config,embedder,monkeypatch):
    legacy_database(config.path,embedder); before=inspect(config.path)
    with monkeypatch.context() as patch:
        def fail(_): raise sqlite3.OperationalError("synthetic-sensitive")
        patch.setattr(cache_schema,"create_expiration_index",fail)
        with pytest.raises(DatabaseError) as caught: open_cache(embedder,database_config=config)
        assert "synthetic" not in str(caught.value)
    assert inspect(config.path)==before
    with open_cache(embedder,database_config=config): pass
    assert verify_database(config)["connection"]=="Verified"

def test_lock_failure_recovery_and_timeout(config,embedder):
    initialize_database(config)
    with closing(sqlite3.connect(config.path)) as lock:
        lock.execute("BEGIN EXCLUSIVE")
        with pytest.raises(DatabaseError): open_cache(embedder,database_config=config)
        with pytest.raises(DatabaseError): verify_database(config)
        lock.rollback()
    with open_cache(embedder,database_config=config) as cache: save(cache)
    assert verify_database(config)["connection"]=="Verified"

@pytest.mark.parametrize("kind",["unrelated","future","corrupt","missing_index"])
def test_incompatible_database_never_reset(config,embedder,kind):
    if kind=="corrupt": config.path.write_bytes(b"synthetic corruption")
    else:
        initialize_database(config)
        with closing(sqlite3.connect(config.path)) as db, db:
            if kind=="unrelated": db.execute("ALTER TABLE cache_entries ADD COLUMN unexpected TEXT")
            elif kind=="future": db.execute("PRAGMA user_version=99")
            else: db.execute("DROP INDEX cache_expiration")
    before=config.path.read_bytes()
    with pytest.raises(DatabaseError): open_cache(embedder,database_config=config)
    with pytest.raises(DatabaseError): verify_database(config)
    assert config.path.read_bytes()==before

def test_uri_characters_in_real_filename_are_encoded(config,embedder):
    config=replace(config,path=config.path.with_name("cache # percent%.sqlite3"))
    initialize_database(config)
    with open_cache(embedder,database_config=config) as cache: save(cache)
    assert verify_database(config)["connection"]=="Verified"

def test_summary_does_not_connect_or_display_path(config,monkeypatch):
    def forbidden(*args,**kwargs): raise AssertionError("Unexpected connection")
    monkeypatch.setattr(sqlite3,"connect",forbidden)
    result=database_summary({"LCR_DATABASE_MODE":"production","LCR_DATABASE_PATH":str(config.path)})
    assert result["connection"]=="Not probed" and result["configuration"]=="Configured"
    assert str(config.path) not in json.dumps(result) and str(config.path) not in repr(config)

def test_parent_directory_is_not_created(config):
    config=replace(config,path=config.path.parent/"missing"/"cache.sqlite3")
    with pytest.raises(DatabaseError): initialize_database(config)
    assert not config.path.parent.exists()

def test_schema_failure_closes_connection(config,embedder,monkeypatch):
    initialize_database(config)
    with closing(sqlite3.connect(config.path)) as db:
        db.execute("DROP INDEX cache_expiration")
    original=sqlite3.connect; connections=[]
    def tracked(*args,**kwargs):
        connection=original(*args,**kwargs); connections.append(connection); return connection
    monkeypatch.setattr(sqlite3,"connect",tracked)
    with pytest.raises(DatabaseError): open_cache(embedder,database_config=config)
    assert len(connections)==1
    with pytest.raises(sqlite3.ProgrammingError): connections[0].execute("SELECT 1")

def test_admin_error_is_sanitized(monkeypatch,capsys):
    import database_admin
    monkeypatch.setattr(database_admin,"load_database_config",lambda:load_database_config({"LCR_DATABASE_PATH":"postgresql://synthetic"}))
    assert database_admin.main(["verify"])==1
    assert "synthetic" not in capsys.readouterr().out
