"""Streamlit presentation only; analytics and model calls remain separate."""

from datetime import datetime, timezone
from html import escape
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import sqlite3

import streamlit as st

from dashboard_data import (
    PROJECT_ROOT, BenchmarkMetadata, RuntimeMetadata, SourceResult,
    load_benchmark, load_runtime,
)


SECTIONS = (
    "Overview", "Cache Analytics", "Model Routing",
    "Cost Analysis", "Performance", "System Information",
)
STATUS_TEXT = {
    "ready": "Available", "empty": "No observations", "missing": "Not found",
    "invalid": "Needs attention", "unreadable": "Unavailable", "too_large": "Limit reached",
}
SECTION_DETAILS = {
    "Cache Analytics": (
        "Understand reuse, eligibility, and the trade-off between hits and false hits.",
        "Cache analytics are coming next",
        "Threshold comparisons and safe-reuse metrics will be added in Task 2. "
        "This workspace keeps benchmark evidence separate from runtime activity.",
        (("Reuse and eligibility", "Explore true hits, false hits, misses, and eligibility decisions."),
         ("Threshold comparisons", "Compare reviewed query pairs across similarity thresholds.")),
    ),
    "Model Routing": (
        "Follow complexity-based selection and quality-based escalation.",
        "Routing analytics are coming next",
        "Initial model selection and escalation will have separate views in Task 2. "
        "The dashboard does not submit prompts or invoke models.",
        (("Initial selection", "Inspect complexity labels and the selected small or large role."),
         ("Quality and escalation", "Trace additional attempts and the final heuristic evaluation.")),
    ),
    "Cost Analysis": (
        "Keep estimated spend, missing pricing, and token usage visible.",
        "Cost analytics are coming next",
        "Task 2 will separate complete estimates from partial subtotals. "
        "Unknown pricing and missing usage will remain unavailable, never zero.",
        (("Cost coverage", "Distinguish known estimates from requests without usable pricing."),
         ("Per-model accounting", "Separate generation and embedding costs by source and model.")),
    ),
    "Performance": (
        "Separate cache lookup time from generation time and local mock overhead.",
        "Performance analytics are coming next",
        "Task 2 will expose timing views with clear measurement provenance. "
        "Offline mock-call timings do not measure Gemini API latency.",
        (("Cache lookup", "Review local embedding, SQLite, and similarity-search timing."),
         ("Generation", "Keep measured runtime timings separate from offline mock execution.")),
    ),
}


def _cards(items):
    """Only fixed copy and sanitized metadata enter these escaped HTML cards."""
    cards = []
    for title, value, description, ready in items:
        value_class = "lcr-value lcr-ready" if ready else "lcr-value"
        cards.append(
            '<article class="lcr-card">'
            f"<h3>{escape(str(title))}</h3>"
            f'<div class="{value_class}">{escape(str(value))}</div>'
            f"<p>{escape(str(description))}</p></article>"
        )
    st.html('<div class="lcr-grid">' + "".join(cards) + "</div>")


def _note(text):
    st.html(f'<div class="lcr-note">{escape(text)}</div>')


def _source_panel(source: SourceResult):
    title = "Benchmark report" if source.kind == "benchmark" else "Request records"
    filename = "benchmark_report.json" if source.kind == "benchmark" else "request_metrics.jsonl"
    with st.container(border=True, key=f"source_{source.kind}"):
        st.markdown(f"**{title}**")
        st.caption(source.label)
        if source.status in ("invalid", "unreadable", "too_large"):
            st.warning(source.message)
        elif source.status in ("missing", "empty"):
            st.info(source.message)
        else:
            st.markdown("**Available** · source metadata validated")
        st.caption(f"Expected local file: {filename}")
        if source.kind == "benchmark":
            st.caption("Synthetic fixtures and mocked model calls. Not live API performance.")
        else:
            st.caption("Local instrumentation records. Their origin is not independently verified.")


def _overview_summary(benchmark, runtime):
    """Show source metadata, never inferred performance or synthetic spending."""
    report = benchmark.metadata
    records = runtime.metadata
    pair_count = report.pair_count if isinstance(report, BenchmarkMetadata) else None
    routing_count = report.routing_query_count if isinstance(report, BenchmarkMetadata) else None
    threshold_count = len(report.thresholds) if isinstance(report, BenchmarkMetadata) else None
    record_count = records.record_count if isinstance(records, RuntimeMetadata) else None

    def value(number):
        return "Unavailable" if number is None else f"{number:,}"

    cards = []
    for label, number, hint, value_class in (
        ("Reviewed query pairs", pair_count, "Offline benchmark · synthetic", "lcr-count"),
        ("Routing queries", routing_count, "Synthetic query fixtures", "lcr-tile-value"),
        ("Threshold settings", threshold_count, "Benchmark configurations", "lcr-tile-value"),
        ("Request records", record_count, STATUS_TEXT[runtime.status], "lcr-tile-value"),
    ):
        cards.append(
            f'<article class="lcr-card"><h3>{escape(label)}</h3>'
            f'<div class="{value_class}">{value(number)}</div><p>{escape(hint)}</p></article>'
        )
    st.html('<div class="lcr-grid lcr-stats">' + "".join(cards) + '</div>')
    with st.container(border=True, key="benchmark_chart"):
        st.subheader("Cache hits by similarity threshold")
        st.caption("Offline benchmark · correct and false answer reuse across reviewed pairs.")
        if isinstance(report, BenchmarkMetadata) and report.hit_counts:
            rows = sorted(zip(report.thresholds, report.hit_counts))
            st.area_chart({
                "Threshold": [t for t, _ in rows],
                "Correct hits": [hits for _, (hits, _) in rows],
                "False hits": [errors for _, (_, errors) in rows],
            }, x="Threshold", y=["Correct hits", "False hits"], stack=False,
                color=["#2a705b", "#90ac42"], y_label="Query pairs", height=270)
        else:
            st.info("The chart needs validated hit counts for every benchmark threshold.")
    latest = records.latest_timestamp_utc if isinstance(records, RuntimeMetadata) else "Unavailable"
    st.html('<div class="lcr-note">Latest recorded request: ' + escape(latest) +
            ' · Model connection: Not connected · Benchmark: ' +
            escape(STATUS_TEXT[benchmark.status]) + '</div>')


def _overview(benchmark, runtime):
    st.title("Workspace overview")
    st.write("A clear view of your data sources and the components behind each decision.")
    _overview_summary(benchmark, runtime)
    st.subheader("Data sources")
    st.caption("Refresh checks the two local files again. Source data is never merged automatically.")
    with st.container(horizontal=True, key="source_grid"):
        _source_panel(benchmark)
        _source_panel(runtime)
    st.subheader("Component map")
    st.caption("Implemented modules are independently usable; the dashboard does not execute this pipeline.")
    _cards([
        ("01 / Reuse", "Semantic cache", "SQLite persistence with explicit eligibility and version scoping.", False),
        ("02 / Select", "Model routing", "Complexity classification selects a configured model role.", False),
        ("03 / Assess", "Answer quality", "Deterministic heuristics with transparent reasons.", False),
        ("04 / Review", "Escalation", "At most one additional large-model call after a failed small answer.", False),
    ])


def _analytics(section, benchmark, runtime):
    description, title, body, coverage = SECTION_DETAILS[section]
    st.title(section)
    st.write(description)
    _note("Workspace foundation · detailed analytics are planned for Task 2.")
    st.html(
        '<section class="lcr-empty" aria-label="Analytics empty state">'
        '<div class="lcr-eyebrow">Prepared workspace</div>'
        f"<h3>{escape(title)}</h3><p>{escape(body)}</p></section>"
    )
    _cards([(heading, "Planned", explanation, False) for heading, explanation in coverage])
    st.subheader("Source readiness")
    _cards([
        ("Offline benchmark", STATUS_TEXT[benchmark.status], "Synthetic evidence, not production performance.", benchmark.status == "ready"),
        ("Request records", STATUS_TEXT[runtime.status], "Local records; no automatic live connection.", runtime.status == "ready"),
    ])
    for source in (benchmark, runtime):
        if source.status in ("invalid", "unreadable", "too_large"):
            st.warning(f"{source.label}: {source.message}")
        elif source.status in ("missing", "empty"):
            st.info(f"{source.label}: {source.message}")


def _package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "Unavailable"


def _system_info(benchmark, runtime):
    st.title("System Information")
    st.write("Local software versions, data access, and application boundaries.")
    _cards([
        ("Python", platform.python_version(), "Current dashboard interpreter", False),
        ("Streamlit", _package_version("streamlit"), "Installed dashboard framework", False),
        ("SQLite", sqlite3.sqlite_version, "Python's built-in database engine", False),
    ])
    st.subheader("Application boundaries")
    with st.container(border=True):
        for heading, detail in (
            ("Dashboard", "Read-only local files, working navigation, and source refresh."),
            ("Existing CLI", "Remains a separate application with its existing history and memory behavior."),
            ("Cache and routing", "Independent modules; no dashboard generation or storage actions."),
            ("Benchmarks", "Offline fixture experiments. Their results do not establish production performance."),
        ):
            st.markdown(f"**{heading}**")
            st.caption(detail)
    st.subheader("Source checks")
    _source_panel(benchmark)
    _source_panel(runtime)
    with st.expander("What the dashboard reads"):
        st.write("Only benchmark_report.json and request_metrics.jsonl are data sources.")
        st.write("The UI receives validated source metadata. Raw reports, query text, responses, "
                 "environment values, and exception details are not displayed.")
        st.write("Sources are limited to 2 MiB each and 10,000 request records. "
                 "Linked files are rejected. No model clients or database connections are created.")
        st.caption("The benchmark loader validates the metadata used here. Detailed metric validation "
                   "and presentation will be extended alongside Task 2.")


def render_dashboard():
    st.set_page_config(
        page_title="LLM Cost Router", page_icon=":material/alt_route:",
        layout="wide", initial_sidebar_state="auto",
    )
    stylesheet = Path(__file__).resolve().parent / "assets" / "dashboard.css"
    try:
        st.html("<style>" + stylesheet.read_text(encoding="utf-8") + "</style>")
    except OSError:
        st.caption("Custom styling is unavailable. Standard dashboard controls remain usable.")
    with st.sidebar:
        st.html(
            '<div class="lcr-brand"><div class="lcr-mark" aria-hidden="true">LCR</div>'
            '<div><strong>LLM Cost Router</strong><span class="lcr-eyebrow">Developer workspace</span></div></div>'
        )
        with st.container(key="workspace_nav"):
            section = st.radio("Workspace navigation", SECTIONS, label_visibility="collapsed",
                               key="dashboard_section", width="stretch")
        st.divider()
        st.caption("LOCAL WORKSPACE / READ ONLY")
        st.caption("Offline benchmarks and local request records have separate identities.")
        st.caption("No model requests are made by this dashboard.")

    with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
        st.caption(f"WORKSPACE / {section.upper()}")
        st.button("Refresh data", icon=":material/refresh:", key="refresh_data",
                  help="Read the local benchmark report and request records again.")
    with st.spinner("Checking local data sources..."):
        benchmark = load_benchmark(PROJECT_ROOT)
        runtime = load_runtime(PROJECT_ROOT)
    checked = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    if section == "Overview":
        _overview(benchmark, runtime)
    elif section == "System Information":
        _system_info(benchmark, runtime)
    else:
        _analytics(section, benchmark, runtime)
    st.divider()
    st.caption(f"Sources checked at {checked} · Dashboard foundation · Detailed analytics in Task 2")
