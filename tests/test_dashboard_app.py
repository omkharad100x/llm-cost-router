"""Offline Streamlit navigation, refresh, import, and local config checks."""

import builtins
import importlib
import json
from pathlib import Path
import socket
import tomllib

import pytest
from streamlit.testing.v1 import AppTest

import dashboard_ui as ui


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No network calls are permitted.")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    # Windows asyncio builds its internal socketpair using loopback TCP.
    # Permit only that local transport; external model/API connections still fail.
    connect = socket.socket.connect
    connect_ex = socket.socket.connect_ex
    def local_connect(sock, address):
        if not isinstance(address, tuple) or address[0] not in ("127.0.0.1", "::1"):
            return forbidden()
        return connect(sock, address)
    def local_connect_ex(sock, address):
        if not isinstance(address, tuple) or address[0] not in ("127.0.0.1", "::1"):
            return forbidden()
        return connect_ex(sock, address)
    monkeypatch.setattr(socket.socket, "connect", local_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", local_connect_ex)
    original = Path.open
    def safe_open(path, *args, **kwargs):
        assert path.name not in {".env", "memory.txt", "secrets.toml", "credentials.toml"}
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", safe_open)


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "PROJECT_ROOT", tmp_path)
    return AppTest.from_file(str(ROOT / "dashboard.py"), default_timeout=20).run()


def html(app):
    return "\n".join(element.proto.body for element in app.get("html"))


def test_dashboard_import_has_no_loading_or_model_side_effects(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name not in {"main", "dotenv", "google.genai", "model_router", "semantic_cache", "escalation"}
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    def forbidden(*args, **kwargs):
        raise AssertionError("Import must not load data.")
    monkeypatch.setattr(ui, "load_benchmark", forbidden)
    monkeypatch.setattr(ui, "load_runtime", forbidden)
    import dashboard
    importlib.reload(dashboard)


def test_overview_has_working_navigation_and_honest_empty_states(app):
    assert not app.exception
    assert app.title[0].value == "Workspace overview"
    assert app.radio[0].options == list(ui.SECTIONS)
    assert "Not found" in html(app) and "Not connected" in html(app)
    assert len(app.button) == 1 and app.button[0].label == "Refresh data"
    assert not app.text_input and not app.chat_input
    assert "synthetic" in html(app)
    assert any("No source file" in item.value for item in app.info)


@pytest.mark.parametrize("section", ui.SECTIONS)
def test_every_section_renders_without_exception(app, section):
    app.radio[0].set_value(section).run()
    assert not app.exception
    expected = "Workspace overview" if section == "Overview" else section
    assert app.title[0].value == expected
    if section in ui.SECTION_DETAILS and section != "Performance":
        assert "coming next" in html(app)
        assert not app.metric  # Detailed performance metrics belong to Task 2.


def test_refresh_recovers_missing_and_malformed_data(app, tmp_path):
    path = tmp_path / "benchmark_report.json"
    path.write_text('{"synthetic-secret": broken}', encoding="utf-8")
    app.button[0].click().run()
    assert not app.exception
    assert any("could not be validated" in item.value for item in app.warning)
    assert "synthetic-secret" not in html(app)
    path.write_bytes((ROOT / "benchmark_report.json").read_bytes())
    app.button[0].click().run()
    assert not app.exception
    assert "Available" in html(app)
    assert not app.warning


def test_report_text_and_unknown_fields_never_reach_ui(app, tmp_path):
    report = json.loads((ROOT / "benchmark_report.json").read_text(encoding="utf-8"))
    report["measurement_notes"] = ["synthetic-secret <script>alert(1)</script>"]
    report["datasets"]["source"] = "synthetic-secret"
    report["configuration"]["api_key"] = "synthetic-secret"
    (tmp_path / "benchmark_report.json").write_text(json.dumps(report), encoding="utf-8")
    for section in ui.SECTIONS:
        app.radio[0].set_value(section).run()
        assert not app.exception
        output = html(app) + str([item.value for item in app.caption])
        assert "synthetic-secret" not in output
        assert "alert(1)" not in output


def test_system_information_has_no_environment_or_host_paths(app):
    app.radio[0].set_value("System Information").run()
    assert not app.exception
    assert "Python" in html(app) and "Streamlit" in html(app)
    assert "GEMINI_API_KEY" not in html(app)
    assert str(ROOT) not in html(app)


def test_theme_and_server_defaults_are_local():
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert config["server"]["address"] == "127.0.0.1"
    assert config["server"]["headless"] is True
    assert config["browser"]["gatherUsageStats"] is False
    assert config["theme"]["base"] == "light"
    assert config["client"]["showErrorDetails"] == "none"


@pytest.mark.parametrize("pair_count,routing_count", [(7, 3), (0, 0)])
def test_overview_counts_follow_validated_metadata(app, monkeypatch, pair_count, routing_count):
    metadata = ui.BenchmarkMetadata(pair_count, routing_count, (0.8, 0.9))
    monkeypatch.setattr(ui, "load_benchmark", lambda _: ui.SourceResult(
        "benchmark", "ready" if pair_count else "empty", "Test metadata.", metadata))
    app.button[0].click().run()
    assert not app.exception
    output = html(app)
    assert f'<div class="lcr-count">{pair_count}</div>' in output
    assert f'<div class="lcr-tile-value">{routing_count}</div>' in output
    assert '<div class="lcr-tile-value">2</div>' in output
    assert "Unavailable" in output  # Absent request records never become zero.


def test_analytics_threshold_selection_and_separate_historical_panel(app, tmp_path):
    report = json.loads((ROOT / "benchmark_report.json").read_bytes())
    (tmp_path / "benchmark_report.json").write_text(json.dumps(report), encoding="utf-8")
    app.radio[0].set_value("Cache Analytics").run()
    assert not app.exception
    for experiment in report["cache_benchmarks"]:
        app.selectbox[0].set_value(experiment["threshold"]).run()
        assert not app.exception
        assert f'{experiment["metrics"]["hit_rate"]:.1%}' in html(app)
        assert "Fixture responses stored at this threshold" in html(app)
        assert len(app.table) == 1
    app.radio[0].set_value("Model Routing").run()
    assert not app.exception and len(app.table) == 1
    assert "Historical · large_model records" in html(app)
    assert "Unavailable" in html(app)
    (tmp_path / "request_metrics.jsonl").write_text("", encoding="utf-8")
    app.button[0].click().run()
    assert not app.exception
    for role in ("Small", "Large"):
        assert f'<h3>{role} model</h3><div class="lcr-value lcr-ready">0</div>' in html(app)

def test_cost_comparison_navigation_and_missing_usage(app, tmp_path):
    from benchmarks import run_benchmarks
    report = run_benchmarks()
    (tmp_path / "benchmark_report.json").write_text(json.dumps(report), encoding="utf-8")
    for section in ui.SECTIONS:
        app.radio[0].set_value(section).run()
        assert not app.exception
    app.radio[0].set_value("Cost Analysis").run()
    assert not app.exception
    output = html(app)
    assert "Baseline cost" in output and "Optimized known subtotal" in output
    assert "Unavailable" in output and "0.001725" in output
    assert any("Token usage unavailable for gemini-embedding-001" in item.value for item in app.info)
    assert len(app.table) == 1
    report.pop("cost_comparison")
    (tmp_path / "benchmark_report.json").write_text(json.dumps(report), encoding="utf-8")
    app.button[0].click().run()
    assert not app.exception and any("no paired cost workload" in item.value for item in app.info)


def test_historical_cost_cards_share_exact_accounting(app, tmp_path):
    from types import SimpleNamespace
    from instrumentation import usage_metrics
    from dashboard_metrics import historical_metrics

    usage = SimpleNamespace(prompt_token_count=1000, candidates_token_count=100,
                            thoughts_token_count=50, total_token_count=1150,
                            cached_content_token_count=500, tool_use_prompt_token_count=0)
    row = dict(timestamp_utc="2026-11-06T00:00:00+00:00", model="gemini-offline-fixture",
               request_kind="small_model", status="success", latency_ms=1,
               **usage_metrics(SimpleNamespace(usage_metadata=usage),
                               {"input_usd_per_million":"0.25", "output_usd_per_million":"1.50",
                                "cached_input_usd_per_million":"0.025"}))
    (tmp_path / "request_metrics.jsonl").write_text(json.dumps(row), encoding="utf-8")
    app.radio[0].set_value("Cost Analysis").run()
    assert not app.exception
    expected = historical_metrics([row])
    output = html(app)
    assert expected["display"]["estimated_cost_usd"] in output
    for component in expected["components"].values():
        assert component["display"]["estimated_cost_usd"] in output
    assert "Historical · API requests" in output
    assert len(app.table) == 1
    row["estimated_cost_usd"] = None
    row["cost_status"] = "usage_unavailable"
    for name in ("input_cost_usd", "cached_input_cost_usd", "output_cost_usd"):
        row[name] = None
    (tmp_path / "request_metrics.jsonl").write_text(json.dumps(row), encoding="utf-8")
    app.button[0].click().run()
    assert not app.exception
    assert "Unavailable" in html(app) and "1 requests with unknown cost" in html(app)
