"""Streamlit presentation only; analytics and model calls remain separate."""

from datetime import datetime, timezone
from decimal import Decimal
from html import escape
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import sqlite3

import streamlit as st

from cache_config import DEFAULT_CONFIG
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


def _metric_cards(items):
    _cards([(label, "Unavailable" if value is None else value, hint, value is not None)
            for label, value, hint in items])


def _percent(rate):
    return None if rate is None else f"{rate:.1%}"


def _benchmark_status(source):
    if source.status in ("invalid", "unreadable", "too_large"):
        st.warning(source.message)
    elif source.status != "ready":
        st.info(source.message)
    st.caption("Synthetic benchmark · missing report fields and rates with no observations display as unavailable.")


def _cache_analytics(source):
    st.title("Cache Analytics")
    _benchmark_status(source)
    experiments = (source.analytics or {}).get("cache", [])
    cache = {}
    if experiments:
        thresholds = [c["t"] for c in experiments]
        default = DEFAULT_CONFIG.similarity_threshold
        selected = st.selectbox("Benchmark similarity threshold", thresholds,
                                index=thresholds.index(default) if default in thresholds else 0)
        cache = next(c for c in experiments if c["t"] == selected)
    _metric_cards([
        ("Cache hits", cache.get("hits"), "First lookup per reviewed pair"),
        ("Cache misses", cache.get("misses"), "First lookup per reviewed pair"),
        ("Hit rate", _percent(cache.get("hitRate")), "Hits / reviewed pairs"),
        ("Miss rate", _percent(cache.get("missRate")), "Misses / reviewed pairs"),
        ("Cached responses", cache.get("seeded"), "Fixture responses stored at this threshold"),
        ("Generation calls avoided", cache.get("avoided"), "Includes repeated lookups and false hits"),
        ("Total lookups", cache.get("lookups"), "All benchmark lookup repetitions"),
        ("Average cache lookup", None if cache.get("lookup") is None else f"{cache['lookup']:.2f} ms",
         "SQLite, mock embedding and search; excludes generation"),
    ])
    with st.container(border=True):
        st.subheader("False matches by threshold")
        st.caption("Synthetic benchmark · false hits reuse an answer across different tasks; "
                   "false misses skip an equivalent pair. Each reviewed pair counts once.")
        if any(c["falseHits"] is not None or c["falseMisses"] is not None for c in experiments):
            st.bar_chart({"Threshold": [str(c["t"]) for c in experiments],
                          "False hits": [c["falseHits"] for c in experiments],
                          "False misses": [c["falseMisses"] for c in experiments]},
                         x="Threshold", y=["False hits", "False misses"], stack=False,
                         y_label="Reviewed pairs", color=["#2a705b", "#90ac42"])
        else:
            st.info("The report has no false-hit or false-miss counts.")
        if experiments:
            st.table([{"Threshold": c["t"], "Hits": c["hits"], "Misses": c["misses"],
                       "False hits": c["falseHits"], "False misses": c["falseMisses"]}
                      for c in experiments], hide_index=True)
        if cache:
            false_hits = _percent(cache["falseHitRate"]) or "Unavailable"
            false_misses = _percent(cache["falseMissRate"]) or "Unavailable"
            st.caption(f"Selected threshold · false-hit rate among must-not-match pairs: {false_hits}. "
                       f"False-miss rate among safe-match pairs: {false_misses}.")
    _metric_cards([("Current cached responses", None, "Database inventory unmeasured"),
                   ("Cache storage usage", None, "Database size unmeasured")])
    _note("Historical cache analytics unavailable. Request records omit cache decisions, "
          "timings and avoided calls. Current cached responses and cache storage usage "
          "are unavailable because neither source measures database inventory or bytes.")


def _routing_analytics(source, runtime):
    st.title("Model Routing")
    _benchmark_status(source)
    b = source.analytics or {}
    _metric_cards([
        ("Escalations", b.get("escalations"), "Queries that triggered a second attempt"),
        ("Escalation rate", _percent(b.get("escalationRate")), "Escalations / benchmark queries"),
        ("Initial answer acceptance", _percent(b.get("acceptanceRate")), "First answers passing evaluation / queries"),
        ("Escalation success", _percent(b.get("successRate")), "Passing second answers / triggered escalations"),
    ])
    st.subheader("Query complexity")
    _metric_cards([("Simple", b.get("simple"), "Synthetic benchmark · classified queries"),
                   ("Complex", b.get("complex"), "Synthetic benchmark · classified queries")])
    with st.container(border=True):
        st.subheader("Model selection and usage")
        st.caption("Synthetic benchmark · initial and final selections count queries. "
                   "Model usage includes escalation calls. Final selection uses the returned answer.")
        attempts, final = b.get("attempts", {}), b.get("final", {})
        stages = [{"Stage": "Initial selection", "Small": b.get("small"), "Large": b.get("large")},
                  {"Stage": "All attempts", "Small": attempts.get("small"), "Large": attempts.get("large")},
                  {"Stage": "Final selection", "Small": final.get("small"), "Large": final.get("large")}]
        if any(row[role] is not None for row in stages for role in ("Small", "Large")):
            st.bar_chart(stages, x="Stage", y=["Small", "Large"], stack=False,
                         color=["#2a705b", "#90ac42"], y_label="Count")
            st.table(stages, hide_index=True)
        else:
            st.info("Model selection and attempt counts are unavailable in this report.")
    st.subheader("Recorded model attempts")
    records = runtime.metadata
    if isinstance(records, RuntimeMetadata):
        small, large = records.small_attempts, records.large_attempts
    elif runtime.status == "empty":
        small, large = 0, 0
    else:
        small, large = None, None
        st.info(runtime.message)
    _metric_cards([("Small model", small, "Historical · small_model records"),
                   ("Large model", large, "Historical · large_model records")])
    _note("Historical counts include successful and failed role-tagged calls. Request records "
          "omit query classifications, attempt links, quality decisions and final selections.")


def _usd(value):
    return None if value is None else f"$\u2009{Decimal(value):,.8f}"


def _cost_analysis(source, runtime):
    st.title("Cost Analysis")
    comparison = (source.analytics or {}).get("comparison", {})
    if comparison.get("status") != "ready":
        st.info(" ".join(comparison.get("issues", ["The benchmark report is unavailable."])))
        _metric_cards([(name, None, "Paired benchmark required") for name in
                       ("Baseline cost", "Optimized cost", "Estimated savings", "Savings percentage")])
    else:
        st.caption(f"Synthetic benchmark estimates · USD · {comparison['query_count']} paired queries · "
                   f"cache threshold {comparison['threshold']}")
        base, optimized = comparison["baseline"], comparison["optimized"]
        _metric_cards([
            ("Baseline cost", _usd(base["estimated_cost_usd"]), "One large-model generation per query"),
            ("Optimized cost", _usd(optimized["estimated_cost_usd"]), "All generation and embedding attempts"),
            ("Estimated savings", _usd(comparison["savings_usd"]), "Withheld for incomplete costs or unsafe reuse"),
            ("Savings percentage", None if comparison["savings_percentage"] is None else
             f"{Decimal(comparison['savings_percentage']):.2f}%", "Savings / baseline cost"),
            ("Baseline cost per query", _usd(base["average_cost_per_query_usd"]), "All paired queries"),
            ("Optimized cost per query", _usd(optimized["average_cost_per_query_usd"]), "Includes cache hits"),
        ])
        for issue in comparison["issues"]:
            st.info(issue)
        with st.container(border=True):
            st.subheader("Scenario comparison")
            st.caption("Generation calls include escalation. Safe cache hits avoid first generation calls.")
            st.bar_chart({"Scenario": ["Baseline", "Optimized"],
                          "Generation calls": [base["generation_calls"], optimized["generation_calls"]]},
                         x="Scenario", y="Generation calls", color="#2a705b")
            if base["complete"] and optimized["complete"]:
                st.bar_chart({"Scenario": ["Baseline", "Optimized"],
                              "Estimated USD": [float(base["estimated_cost_usd"]), float(optimized["estimated_cost_usd"])]},
                             x="Scenario", y="Estimated USD", color="#2a705b")
            else:
                st.caption("The cost chart requires complete estimates for both scenarios.")
        _metric_cards([
            ("Baseline generation calls", base["generation_calls"], "One attempt per query"),
            ("Optimized generation calls", optimized["generation_calls"], "Includes escalations"),
            ("Safe generation calls avoided", comparison["safe_calls_avoided"], "Reviewed cache hits"),
            ("Unsafe cache hits", comparison["cache_outcomes"]["false_hit"], "Blocks all monetary savings claims"),
            ("Baseline known subtotal", _usd(base["known_cost_subtotal_usd"]), f'{base["unknown_cost_requests"]} unpriced requests'),
            ("Optimized known subtotal", _usd(optimized["known_cost_subtotal_usd"]), f'{optimized["unknown_cost_requests"]} unpriced requests'),
            ("Escalation expenditure", _usd(comparison["escalation"]["estimated_cost_usd"]),
             "Known subtotal " + _usd(comparison["escalation"]["known_cost_subtotal_usd"])),
            ("Embedding expenditure", _usd(comparison["embedding"]["estimated_cost_usd"]),
             "Includes lookup and cache storage; known subtotal " + _usd(comparison["embedding"]["known_cost_subtotal_usd"])),
        ])
        st.subheader("Cost by model")
        st.table([{"Model": row["model"], "Scenario": name.title(),
                   "Estimated USD": _usd(row[name]["estimated_cost_usd"]) or "Unavailable",
                   "Known subtotal USD": _usd(row[name]["known_cost_subtotal_usd"]),
                   "Unpriced requests": row[name]["unknown_cost_requests"]}
                  for row in comparison["by_model"] for name in ("baseline", "optimized")], hide_index=True)
        with st.expander("Comparison methodology"):
            st.write("Both scenarios process the same ordered routing fixtures. Each role uses its own "
                     "synthetic response and token-usage fixture, including thinking tokens. "
                     "The optimized run starts with an empty cache and records lookup, storage, generation "
                     "and escalation costs. Query and response eligibility are explicitly reviewed.")
            st.write("Saved attempts use the pricing snapshot from the benchmark run. Missing rates or "
                     "usage withhold totals. False or unreviewed cache hits withhold savings even when costs "
                     "are complete. Synthetic vectors and responses do not establish production quality or savings.")
    st.subheader("Historical and live comparison")
    st.caption("Historical source · " + STATUS_TEXT[runtime.status])
    _note("Historical request logs lack a paired large-model baseline and linked cache decisions. "
          "Historical savings are unavailable. Live measurements require a connected request source.")


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
        st.caption("Benchmark analytics use numeric observations from the saved offline report.")


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
    elif section == "Cache Analytics":
        _cache_analytics(benchmark)
    elif section == "Model Routing":
        _routing_analytics(benchmark, runtime)
    elif section == "Cost Analysis":
        _cost_analysis(benchmark, runtime)
    elif section == "System Information":
        _system_info(benchmark, runtime)
    else:
        _analytics(section, benchmark, runtime)
    st.divider()
    st.caption(f"Sources checked at {checked} · Read-only local dashboard")
