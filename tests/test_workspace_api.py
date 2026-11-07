"""HTTP contract tests for the public Vercel entry point, with no network."""
import io
import json
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys

import pytest

from api import workspace as endpoint
from frontend import workspace_data as projection


ROOT = Path(__file__).resolve().parents[1]
SENTINEL = "private-fixture-must-not-leave-server"


class Request:
    """BaseHTTPRequestHandler socket transport backed by memory."""
    def __init__(self, method, target):
        self.input = io.BytesIO(f"{method} {target} HTTP/1.0\r\nHost: localhost\r\n\r\n".encode())
        self.output = bytearray()

    def makefile(self, *_args):
        return self.input

    def sendall(self, data):
        self.output.extend(data)


def call(method="GET", target="/api/workspace"):
    request = Request(method, target)
    endpoint.handler(request, ("127.0.0.1", 1), object())
    header, body = bytes(request.output).split(b"\r\n\r\n", 1)
    lines = header.decode("ascii").split("\r\n")
    headers = dict(line.split(": ", 1) for line in lines[1:])
    assert headers["Cache-Control"] == "no-store"
    assert headers["Content-Type"].startswith("application/json")
    assert headers["X-Content-Type-Options"] == "nosniff"
    return int(lines[0].split()[1]), headers, body


@pytest.fixture(autouse=True)
def sandbox(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("The public endpoint attempted private history, database or network access.")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(sqlite3, "connect", forbidden)
    monkeypatch.setattr(projection.ds, "load_runtime", forbidden)
    monkeypatch.setattr(projection.ds, "PROJECT_ROOT", tmp_path)
    original = Path.open
    def safe(path, *args, **kwargs):
        assert path.name not in {".env", "memory.txt", "secrets.toml"}
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", safe)


def test_bundled_benchmark_matches_react_contract_without_private_content(tmp_path, monkeypatch):
    report = json.loads((ROOT/"benchmark_report.json").read_bytes())
    report["datasets"]["source"] = SENTINEL
    report["pipeline_benchmark"]["cases"][0]["query"] = SENTINEL
    report["cost_comparison"]["queries"][0]["id"] = SENTINEL
    (tmp_path/"benchmark_report.json").write_text(json.dumps(report), encoding="utf-8")
    (tmp_path/"request_metrics.jsonl").write_text(SENTINEL, encoding="utf-8")
    monkeypatch.setenv("LCR_DATABASE_BACKEND", "postgresql")
    monkeypatch.setenv("LCR_POSTGRES_DSN", "postgresql://worker:"+SENTINEL+"@invalid.test/db")
    status, headers, body = call(target="/api/workspace?from=2026-11-01&to=2026-11-07&model=All+models&currency=USD&rate=1")
    assert status == 200 and int(headers["Content-Length"]) == len(body)
    data = json.loads(body)
    assert set(data) == {"history","costs","database","hs","bench","bs","small","large",
                         "schema","version","threshold","evaluation"}
    assert data["bs"] == "Ready" and data["hs"] == "Unavailable"
    assert data["history"] is data["costs"] is None
    assert data["bench"]["comparison"]["baseline"]["display"]["estimated_cost_usd"] == "$0.0027"
    assert data["bench"]["comparison"]["savings_usd"] is None
    assert data["bench"]["performance"]["source"].startswith("synthetic")
    assert data["bench"]["cache"] and data["bench"]["queries"] > 0
    assert data["database"]["backend"] == "PostgreSQL" and data["database"]["connection"] == "Not probed"
    for private in (SENTINEL, str(tmp_path), "LCR_POSTGRES_DSN", "postgresql://", "cached_response"):
        assert private not in body.decode()


@pytest.mark.parametrize("raw,source", [(None,"Missing"), ("","Empty"), ("{broken","Invalid")])
def test_unavailable_source_preserves_workspace(tmp_path, monkeypatch, raw, source):
    if raw is not None:
        (tmp_path/"benchmark_report.json").write_text(raw, encoding="utf-8")
    monkeypatch.setenv("LCR_DATABASE_BACKEND", "postgresql")
    monkeypatch.delenv("LCR_POSTGRES_DSN", raising=False)
    status, _, body = call()
    data = json.loads(body)
    assert status == 200 and data["bs"] == source and data["bench"] is None
    assert data["database"]["configuration"] == "Invalid" and data["database"]["connection"] == "Not probed"


@pytest.mark.parametrize("method", ["POST","PUT","PATCH","DELETE","OPTIONS","TRACE","PRIVATE"])
def test_mutations_and_unknown_methods_rejected_before_projection(monkeypatch, method):
    monkeypatch.setattr(endpoint, "workspace", lambda *_a, **_kw:pytest.fail("Unexpected projection"))
    status, headers, body = call(method, "/api/workspace?private="+SENTINEL)
    assert status == 405 and headers["Allow"] == "GET, HEAD"
    assert json.loads(body) == {"error":"Method not allowed."}


@pytest.mark.parametrize("query", ["path=secret", "model=a&model=b", "currency", "model=%ff",
                                    "model="+"a"*129, "rate="+"1"*2050])
def test_bad_filters_are_bounded_and_never_echoed(query):
    status, _, body = call(target="/api/workspace?"+query)
    assert status == 400 and json.loads(body) == {"error":"Invalid workspace filters."}


def test_head_and_safe_errors(monkeypatch, capsys):
    status, headers, body = call("HEAD")
    assert status == 200 and body == b"" and int(headers["Content-Length"]) > 0
    def fail(*_args, **_kwargs):
        raise RuntimeError(SENTINEL)
    monkeypatch.setattr(endpoint, "workspace", fail)
    status, _, body = call(target="/api/workspace?model="+SENTINEL)
    assert status == 503 and json.loads(body) == {"error":"Workspace unavailable."}
    assert SENTINEL not in capsys.readouterr().err


def test_function_import_and_projection_with_site_packages_disabled():
    # Independent runtime smoke test: no installed Streamlit, SDK, Psycopg or Node.
    code = ("import sys; sys.path.insert(0,sys.argv[1]); from api.workspace import workspace; "
            "d=workspace({},include_local_history=False); "
            "assert d['bs']=='Ready' and d['hs']=='Unavailable'")
    subprocess.run([sys.executable,"-I","-S","-B","-c",code,str(ROOT)], check=True,
                   capture_output=True, timeout=15, env={})
