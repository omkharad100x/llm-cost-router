"""Offline isolation and recovery tests with real SQLite connections."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock
import socket
import sqlite3
import numpy as np
import pytest
import semantic_cache
from semantic_cache import SemanticCache, CacheError, CacheConflictError
from embeddings import EmbeddingGenerator, EmbeddingResponseError, MODEL
from instrumentation import InstrumentedGenerator
from escalation import EscalationError
from test_semantic_cache import save, lookup, vector
from test_escalation import setup_pipeline, sdk_response, QUERY

@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args,**kwargs):
        raise AssertionError("Network forbidden.")
    monkeypatch.setattr(socket,"create_connection",denied)
    monkeypatch.setattr(socket.socket,"connect",denied)
    monkeypatch.setattr(socket.socket,"connect_ex",denied)
    original=Path.open
    def safe(path,*args,**kwargs):
        assert path.name not in (".env","memory.txt","secrets.toml","credentials.toml")
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",safe)

def embedder():
    return SimpleNamespace(embed_query=lambda q:vector())

def snapshot(path):
    with closing(sqlite3.connect(path)) as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0]=="ok"
        return db.execute("SELECT * FROM cache_entries ORDER BY id").fetchall()

@pytest.mark.parametrize("namespace",["second-app","' OR 1=1 --","default ","DEFAULT"])
def test_identical_vectors_remain_namespace_isolated(tmp_path,namespace):
    path=tmp_path/"cache.db"
    with SemanticCache(path,embedder()) as a, SemanticCache(path,embedder(),namespace=namespace) as b:
        first=save(a,answer="First answer")
        assert not lookup(b).hit
        second=save(b,answer="Second answer")
        assert first!=second
        assert lookup(a,"Similar query").response=="First answer"
        assert lookup(b,"Similar query").response=="Second answer"
        assert b.invalidate(entry_id=first)==0
        assert b.clear()==1 and lookup(a).entry_id==first

def test_same_namespace_is_shared_not_authenticated(tmp_path):
    path=tmp_path/"cache.db"
    with SemanticCache(path,embedder()) as a, SemanticCache(path,embedder()) as b:
        assert lookup(b).hit is False
        entry=save(a)
        assert lookup(b).entry_id==entry

@pytest.mark.parametrize("mode",["duplicate","conflict","different_keys","different_namespaces"])
def test_concurrent_writes_are_atomic(tmp_path,mode):
    path=tmp_path/"cache.db"
    with SemanticCache(path,embedder()): pass
    gate=Barrier(2,timeout=10)
    def worker(index):
        query=f"query-{index}" if mode=="different_keys" else "same-query"
        answer="same-answer" if mode=="duplicate" else f"answer-{index}"
        namespace=f"app-{index}" if mode=="different_namespaces" else "default"
        def embed(_):
            gate.wait()
            v=np.zeros(768); v[index if mode=="different_keys" else 0]=1
            return v
        with SemanticCache(path,SimpleNamespace(embed_query=embed),namespace=namespace) as cache:
            try:
                return ("saved",save(cache,query=query,answer=answer),query,answer,namespace)
            except CacheConflictError:
                return ("conflict",)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,i) for i in range(2)]
        results=[f.result(timeout=20) for f in futures]
    rows=snapshot(path)
    if mode=="duplicate":
        assert results[0][1]==results[1][1] and len(rows)==1
    elif mode=="conflict":
        assert sorted(r[0] for r in results)==["conflict","saved"] and len(rows)==1
    else:
        assert len(rows)==2
    with closing(sqlite3.connect(path)) as db:
        for r in results:
            if r[0]=="saved":
                assert db.execute("SELECT original_query,cached_response,namespace FROM cache_entries WHERE id=?",(r[1],)).fetchone()==r[2:]

def test_concurrent_reads_with_separate_connections(tmp_path):
    path=tmp_path/"cache.db"
    with SemanticCache(path,embedder()) as cache: entry=save(cache)
    gate=Barrier(4,timeout=10)
    def read(_):
        with SemanticCache(path,embedder()) as cache:
            gate.wait()
            return lookup(cache).entry_id
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(read,range(4)))==[entry]*4

@pytest.mark.parametrize("operation",["save","clear","invalidate","cleanup","read"])
def test_lock_errors_and_recovery_preserve_data(tmp_path,operation):
    path=tmp_path/"cache.db"
    with SemanticCache(path,embedder(),clock=lambda:1000) as cache, closing(sqlite3.connect(path)) as lock:
        entry=save(cache); before=snapshot(path)
        cache._db().execute("PRAGMA busy_timeout=0")
        lock.execute("BEGIN EXCLUSIVE")
        action={"save":lambda:save(cache,query="new"),"clear":cache.clear,
                "invalidate":lambda:cache.invalidate(entry_id=entry),
                "cleanup":cache.remove_expired,"read":lambda:lookup(cache)}[operation]
        try:
            with pytest.raises(CacheError): action()
        finally:
            lock.rollback()
        assert not cache._db().in_transaction and snapshot(path)==before
        assert lookup(cache).entry_id==entry
        assert save(cache,query="new")!=entry
    with SemanticCache(path,embedder(),clock=lambda:1000) as cache: assert lookup(cache).hit
    assert len(snapshot(path))==2

def test_read_failure_is_not_a_miss_and_recovers(tmp_path):
    with SemanticCache(tmp_path/"cache.db",embedder()) as cache:
        entry=save(cache)
        cache._db().set_authorizer(lambda action,*args:sqlite3.SQLITE_DENY if action==sqlite3.SQLITE_READ else sqlite3.SQLITE_OK)
        with pytest.raises(CacheError,match="lookup failed"): lookup(cache)
        cache._db().set_authorizer(None)
        assert lookup(cache).entry_id==entry

def test_uncommitted_write_invisible_and_abandoned_transaction_rolls_back(tmp_path):
    path=tmp_path/"cache.db"
    with SemanticCache(path,embedder()) as cache:
        entry=save(cache,answer="Committed")
        writer=sqlite3.connect(path)
        try:
            writer.execute("BEGIN IMMEDIATE")
            writer.execute("UPDATE cache_entries SET cached_response=? WHERE id=?",("Uncommitted",entry))
            assert lookup(cache).response=="Committed"
        finally:
            writer.close()
    with SemanticCache(path,embedder()) as cache: assert lookup(cache).response=="Committed"
    snapshot(path)

def test_failed_commit_rolls_back_expired_replacement(tmp_path):
    path=tmp_path/"cache.db"; now=[1000]
    with SemanticCache(path,embedder(),clock=lambda:now[0]) as cache:
        entry=save(cache,ttl_seconds=1); before=snapshot(path); now[0]=1002
        def auth(action,name,*args):
            return sqlite3.SQLITE_DENY if action==sqlite3.SQLITE_TRANSACTION and name=="COMMIT" else sqlite3.SQLITE_OK
        cache._db().set_authorizer(auth)
        with pytest.raises(CacheError,match="storage failed"): save(cache)
        cache._db().set_authorizer(None)
        assert not cache._db().in_transaction and snapshot(path)==before
        assert not lookup(cache).hit
        assert save(cache)!=entry

def test_identifier_collision_never_replaces_another_query(tmp_path,monkeypatch):
    path=tmp_path/"cache.db"
    with SemanticCache(path,embedder()) as cache:
        entry=save(cache); before=snapshot(path)
        monkeypatch.setattr(semantic_cache.uuid,"uuid4",lambda:SimpleNamespace(hex=entry))
        with pytest.raises(CacheError): save(cache,query="different",answer="different")
        assert snapshot(path)==before and lookup(cache).entry_id==entry

@pytest.mark.parametrize("operation",["lookup","save"])
@pytest.mark.parametrize("failure",[TimeoutError("synthetic"),ConnectionError("synthetic"),RuntimeError("synthetic")])
def test_embedding_failure_observable_and_nonmutating(tmp_path,operation,failure):
    path=tmp_path/"cache.db"
    with SemanticCache(path,embedder()) as cache: save(cache)
    before=snapshot(path); client=Mock(); records=[]
    client.models.embed_content.side_effect=failure
    generator=InstrumentedGenerator(client,{MODEL:{"input_usd_per_million":"0.15"}},records.append)
    with SemanticCache(path,EmbeddingGenerator(generator)) as cache:
        with pytest.raises(type(failure)):
            lookup(cache) if operation=="lookup" else save(cache,query="new")
    assert snapshot(path)==before
    assert len(records)==client.models.embed_content.call_count==1
    assert records[0]["status"]=="error"
    assert records[0]["input_tokens"] is records[0]["estimated_cost_usd"] is None
    client.models.generate_content.assert_not_called()

@pytest.mark.parametrize("values",[None,[],[1,0],[0]*768,[float("nan")]*768])
def test_invalid_embeddings_cannot_store_or_hit(tmp_path,values):
    client=Mock(); records=[]
    client.models.embed_content.return_value=SimpleNamespace(embeddings=[SimpleNamespace(values=values)])
    with SemanticCache(tmp_path/"cache.db",EmbeddingGenerator(InstrumentedGenerator(client,{},records.append))) as cache:
        with pytest.raises(EmbeddingResponseError): save(cache)
        with pytest.raises(EmbeddingResponseError): lookup(cache)
        assert cache._db().execute("SELECT count(*) FROM cache_entries").fetchone()[0]==0
    assert len(records)==2 and all(r["estimated_cost_usd"] is None for r in records)

def test_two_provider_failures_do_not_trigger_fallback_after_first_error():
    pipeline,client,records=setup_pipeline(replies=[TimeoutError("synthetic"),RuntimeError("synthetic")])
    with pytest.raises(EscalationError) as caught: pipeline.generate(QUERY)
    assert caught.value.stage=="initial_generation"
    assert caught.value.total_attempts==client.models.generate_content.call_count==len(records)==1
    assert records[0]["status"]=="error" and records[0]["estimated_cost_usd"] is None

def test_escalation_timeout_retains_initial_answer_but_raises():
    pipeline,client,records=setup_pipeline(replies=[sdk_response("I do not know."),TimeoutError("synthetic")])
    with pytest.raises(EscalationError) as caught: pipeline.generate(QUERY)
    error=caught.value
    assert error.stage=="escalation_generation" and error.total_attempts==2
    assert error.attempts[0].response.text=="I do not know." and error.attempts[1].response is None
    assert error.totals.estimated_cost_usd is None
    assert [r["status"] for r in records]==["success","error"]
    assert client.models.generate_content.call_count==2

def test_corrupt_candidates_are_observable(tmp_path):
    with SemanticCache(tmp_path/"cache.db",embedder()) as cache:
        broken=save(cache,query="broken"); valid=save(cache,query="valid")
        cache._db().execute("UPDATE cache_entries SET embedding=? WHERE id=?",(b"broken",broken))
        cache._db().commit()
        result=lookup(cache)
        assert result.hit and result.entry_id==valid and result.skipped_corrupt_entries==1
        cache.invalidate(entry_id=valid)
        result=lookup(cache)
        assert not result.hit and result.response is None and result.skipped_corrupt_entries==1


@pytest.mark.parametrize("field,value", [
    ("embedding", b""), ("embedding", "not bytes"), ("original_query", " "),
    ("query_key", "mismatched-key"), ("response_model", ""), ("created_at", "bad-time"),
])
def test_malformed_records_are_reported_and_never_hit(tmp_path,field,value):
    with SemanticCache(tmp_path/"cache.db",embedder()) as cache:
        entry=save(cache)
        # Confirm the normal SQL constraint first, then simulate externally
        # corrupted storage that must still be rejected when decoding.
        if field == "created_at":
            with pytest.raises(sqlite3.IntegrityError):
                cache._db().execute("UPDATE cache_entries SET created_at=? WHERE id=?",(value,entry))
            cache._db().rollback()
            cache._db().execute("PRAGMA ignore_check_constraints=ON")
        # Field names are controlled test constants, values remain parameters.
        cache._db().execute(f"UPDATE cache_entries SET {field}=? WHERE id=?",(value,entry))
        cache._db().commit()
        result=lookup(cache)
        assert not result.hit and result.response is None
        assert result.skipped_corrupt_entries==1

def test_unavailable_database_path_fails_before_embedding(tmp_path):
    client=Mock()
    with pytest.raises(CacheError,match="initialization failed"):
        SemanticCache(tmp_path/"missing-directory"/"cache.db",client)
    client.embed_query.assert_not_called()

def test_migration_failure_then_retry_preserves_legacy_rows(tmp_path,monkeypatch):
    import cache_schema
    from test_cache_versioning import legacy_database, inspect
    path=tmp_path/"legacy.db"
    emb=Mock(); emb.embed_query.return_value=vector()
    legacy_database(path,emb)
    before=inspect(path)
    def fail(_):
        raise sqlite3.OperationalError("synthetic index failure")
    with monkeypatch.context() as patch:
        patch.setattr(cache_schema,"create_expiration_index",fail)
        with pytest.raises(CacheError):
            SemanticCache(path,emb,clock=lambda:1000)
    assert inspect(path)==before
    with SemanticCache(path,emb,clock=lambda:1000):
        pass
    version,rows=inspect(path)
    assert version==2 and len(rows)==len(before[1])
    for old,new in zip(before[1],rows):
        assert {key:new[key] for key in old}==old
    snapshot(path)
