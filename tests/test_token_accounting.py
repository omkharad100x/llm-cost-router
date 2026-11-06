"""Offline hand-calculated token, pricing, precision and dashboard regressions."""

from datetime import date
from decimal import Decimal, localcontext
import importlib.util
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

from benchmark_accounting import summarize_requests
from dashboard_metrics import historical_metrics
from instrumentation import embedding_usage_metrics, load_pricing, usage_metrics
from money import add, amount, display, divide, multiply


ROOT = Path(__file__).resolve().parents[1]
RATES = {"input_usd_per_million": "0.25", "output_usd_per_million": "1.50",
         "cached_input_usd_per_million": "0.025"}
BASE_USAGE = dict(prompt_token_count=1000, candidates_token_count=100, thoughts_token_count=50,
                  total_token_count=1150, cached_content_token_count=0, tool_use_prompt_token_count=0)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("No external calls permitted.")
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket.socket, "connect", denied)
    original = Path.open
    def safe(path, *args, **kwargs):
        assert path.name not in {".env", "memory.txt", "secrets.toml"}
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", safe)


def generation(**changes):
    return SimpleNamespace(usage_metadata=SimpleNamespace(**(BASE_USAGE | changes)))


def record(**changes):
    return dict(timestamp_utc="2026-11-06T00:00:00+00:00", model="gemini-test",
                request_kind="small_model", status="success", latency_ms=2,
                **usage_metrics(generation(**changes), RATES))


@pytest.mark.parametrize("input_count,output_count,thoughts,expected", [
    (1000, 0, 0, "0.00025"), (0, 100, 0, "0.00015"),
    (1000, 100, 50, "0.000475"), (0, 0, 0, "0"),
])
def test_input_output_thinking_and_zero(input_count, output_count, thoughts, expected):
    result = usage_metrics(generation(prompt_token_count=input_count, candidates_token_count=output_count,
        thoughts_token_count=thoughts, total_token_count=input_count+output_count+thoughts), RATES)
    assert Decimal(result["estimated_cost_usd"]) == Decimal(expected)
    assert result["billed_output_tokens"] == output_count + thoughts
    assert add(result[k] for k in ("input_cost_usd","cached_input_cost_usd","output_cost_usd")) == Decimal(expected)


def test_cached_input_is_subtracted_before_ordinary_rate():
    result = usage_metrics(generation(cached_content_token_count=500, thoughts_token_count=0,
                                       total_token_count=1100), RATES)
    assert Decimal(result["input_cost_usd"]) == Decimal("0.000125")
    assert Decimal(result["cached_input_cost_usd"]) == Decimal("0.0000125")
    assert Decimal(result["output_cost_usd"]) == Decimal("0.00015")
    assert Decimal(result["estimated_cost_usd"]) == Decimal("0.0002875")


def test_unreported_cache_split_is_unknown():
    result = usage_metrics(generation(cached_content_token_count=None), RATES)
    assert result["cached_input_tokens"] is result["input_cost_usd"] is result["estimated_cost_usd"] is None
    assert Decimal(result["output_cost_usd"]) == Decimal("0.000225")


def test_zero_prompt_cannot_contain_cached_tokens():
    result = usage_metrics(generation(prompt_token_count=0,cached_content_token_count=None,total_token_count=150),RATES)
    assert result["cached_input_tokens"] is None
    assert Decimal(result["estimated_cost_usd"]) == Decimal("0.000225")


@pytest.mark.parametrize("field", ["prompt_token_count","candidates_token_count"])
def test_missing_required_usage_never_becomes_zero(field):
    result = usage_metrics(generation(**{field:None}), RATES)
    assert result["estimated_cost_usd"] is None
    if field=="prompt_token_count":
        assert Decimal(result["output_cost_usd"]) == Decimal("0.000225")


@pytest.mark.parametrize("field",list(BASE_USAGE))
@pytest.mark.parametrize("value",[-1, True, 1.5])
def test_malformed_sdk_counters_cannot_be_treated_as_omitted(field,value):
    result = usage_metrics(generation(**{field:value}), RATES)
    assert result["estimated_cost_usd"] is None
    assert result["cost_status"] == "usage_inconsistent"


def test_tools_retained_without_guessing_their_billing():
    result = usage_metrics(generation(tool_use_prompt_token_count=12,total_token_count=1162),RATES)
    assert result["tool_use_tokens"] == 12
    assert result["estimated_cost_usd"] is result["billed_output_tokens"] is None


def test_missing_thoughts_inferred_only_as_billable_output():
    result = usage_metrics(generation(thoughts_token_count=None),RATES)
    assert result["thinking_tokens"] is None and result["billed_output_tokens"] == 150
    assert Decimal(result["estimated_cost_usd"]) == Decimal("0.000475")


def test_missing_price_and_missing_cache_rate():
    missing = usage_metrics(generation(),None)
    assert missing["cost_status"] == "pricing_unavailable" and missing["input_tokens"] == 1000
    rates = {k:v for k,v in RATES.items() if k!="cached_input_usd_per_million"}
    result = usage_metrics(generation(cached_content_token_count=500),rates)
    assert result["estimated_cost_usd"] is None and result["cost_status"]=="cached_pricing_unavailable"
    assert result["cached_input_cost_usd"] is None


@pytest.mark.parametrize("count,expected",[(10,"0.0000015"),(0,"0"),(None,None),(2**53+2.0,None)])
def test_embedding_statistics_are_independent(count,expected):
    response=SimpleNamespace(embeddings=[SimpleNamespace(statistics=SimpleNamespace(token_count=count))])
    result=embedding_usage_metrics(response,{"input_usd_per_million":"0.15"})
    if expected is None:
        assert result["estimated_cost_usd"] is None
    else:
        assert Decimal(result["estimated_cost_usd"])==Decimal(expected)
    assert result["output_tokens"] is None and result["output_cost_usd"]=="0"


def test_large_token_counts_keep_fractional_charges():
    count=10**35+1
    result=usage_metrics(generation(prompt_token_count=count,candidates_token_count=0,
                                   thoughts_token_count=0,total_token_count=count),RATES)
    assert Decimal(result["estimated_cost_usd"])==Decimal("25000000000000000000000000000.00000025")
    expected=Decimal("150000000000000000000000000000.000000000000000000000000000001")
    with localcontext() as context:
        context.prec=6
        assert add(["150000000000000000000000000000","0.000000000000000000000000000001"])==expected
        assert multiply("0.123456789123456789","2") == Decimal("0.246913578246913578")
        assert divide("1","3")==Decimal("0.3333333333333333333333333333")


@pytest.mark.parametrize("value,expected", [
    ("0.000000025","$0.00000002"),("0.000000035","$0.00000004"),
    ("0.000000001","<$0.00000001"),("-0.000000001","-<$0.00000001"),
    ("-0.0001","-$0.0001"),("0","$0.00"),(None,None),
])
def test_display_rounds_half_even_without_faking_zero(value,expected):
    assert display(value)==expected


@pytest.mark.parametrize("value",[True,0.1,"NaN","Infinity","1e999"])
def test_invalid_monetary_values_rejected(value):
    with pytest.raises(ValueError):
        amount(value)


def test_verified_prices_and_expiry():
    rates=load_pricing(ROOT/"model_pricing.json",as_of=date(2026,11,6))
    large=rates["gemini-3.8-flash"]
    assert large["input_usd_per_million"]=="0.75"
    assert large["output_usd_per_million"]=="3.75"
    assert large["cached_input_usd_per_million"]=="0.075"
    assert large["verified_on"]=="2026-11-06" and large["source"].startswith("https://ai.google.dev/")
    assert "gemini-3.8-flash" in load_pricing(ROOT/"model_pricing.json",as_of=date(2026,12,31))
    future=load_pricing(ROOT/"model_pricing.json",as_of=date(2027,1,1))
    assert "gemini-3.8-flash" not in future and "gemini-3.1-flash-lite" in future


@pytest.mark.parametrize("value", ["-1",None,True,"NaN"])
def test_invalid_configured_rates_fail_clearly(tmp_path,value):
    path=tmp_path/"rates.json"
    path.write_text(json.dumps({"model":RATES|{"input_usd_per_million":value}}),encoding="utf-8")
    with pytest.raises(ValueError):
        load_pricing(path)


def test_price_json_decimals_never_pass_through_binary_floats(tmp_path):
    path=tmp_path/"rates.json"
    path.write_text('{"model":{"input_usd_per_million":0.123456789123456789,"output_usd_per_million":0}}',encoding="utf-8")
    assert load_pricing(path)["model"]["input_usd_per_million"]=="0.123456789123456789"


def test_unknown_costs_and_component_subtotals_propagate():
    rows=[record(),record(cached_content_token_count=None)]
    result=summarize_requests(rows,query_count=2)
    assert result["cost"]["estimated_cost_usd"] is None
    assert Decimal(result["cost"]["known_cost_subtotal_usd"])==Decimal("0.000475")
    assert result["components"]["input_cost_usd"]["estimated_cost_usd"] is None
    assert Decimal(result["components"]["output_cost_usd"]["estimated_cost_usd"])==Decimal("0.00045")


def test_inconsistent_recorded_components_are_rejected():
    row=record()
    row["input_cost_usd"]="1"
    with pytest.raises(ValueError,match="components"):
        summarize_requests([row],query_count=1)


def test_historical_decimal_filters_components_and_currency():
    rows=[record(),record(cached_content_token_count=500)]
    result=historical_metrics(rows,currency="INR",rate="80.123456789")
    assert Decimal(result["estimated_cost_usd"])==Decimal("0.0008375")
    assert result["display"]["estimated_cost_usd"]=="₹0.0671034"
    assert result["components"]["cached_input_cost_usd"]["display"]["estimated_cost_usd"]=="₹0.00100154"
    assert result["by_model"][0]["estimated_cost_usd"]==result["estimated_cost_usd"]
    empty=historical_metrics(rows,start="2026-11-07")
    assert empty["estimated_cost_usd"]=="0" and empty["average_cost_per_query_usd"] is None


def test_older_records_keep_totals_without_inventing_breakdowns():
    row=record()
    for field in ("input_cost_usd","cached_input_cost_usd","output_cost_usd"):
        row.pop(field)
    result=historical_metrics([row])
    assert Decimal(result["estimated_cost_usd"])==Decimal("0.000475")
    assert all(v["estimated_cost_usd"] is None for v in result["components"].values())


def test_escalation_exact_sum():
    from escalation import _aggregate
    def attempt(cost):
        response=SimpleNamespace(input_tokens=0,output_tokens=0,thinking_tokens=0,
                                 billed_output_tokens=0,latency_ms=0,
                                 estimated_cost_usd=cost,cost_status="estimated")
        return SimpleNamespace(response=response)
    total=_aggregate([attempt("150000000000000000000000000000"),attempt("0.000001")])
    assert Decimal(total.estimated_cost_usd)==Decimal("150000000000000000000000000000.000001")


def test_frontend_adapter_shares_streamlit_accounting(tmp_path,monkeypatch):
    import dashboard_data as data
    spec=importlib.util.spec_from_file_location("workspace_projection",ROOT/"frontend"/"workspace_data.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(data,"PROJECT_ROOT",tmp_path)
    rows=[record(),record(cached_content_token_count=500)]
    (tmp_path/"request_metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows),encoding="utf-8")
    api=module.workspace()
    streamlit=data.load_runtime(tmp_path)
    assert api["costs"]==streamlit.analytics["costs"]
    assert api["history"][0]["cost"]==rows[0]["estimated_cost_usd"]
    assert isinstance(api["history"][0]["cost"],str)
    filtered=module.workspace({"from":"2026-11-07","currency":"INR","rate":"80"})
    assert filtered["costs"]["estimated_cost_usd"]=="0"
    invalid=module.workspace({"from":"bad"})
    assert invalid["costs"] is None and invalid["hs"]=="Ready"


def test_unreported_cache_split_explains_comparison_unavailability():
    from cost_comparison import comparison_metrics

    baseline = record() | {"phase":"baseline", "request_kind":"large_model", "model":"gemini-large"}
    optimized = record(cached_content_token_count=None) | {"phase":"initial", "model":"gemini-small"}
    ledger = dict(schema_version=1, methodology="paired-routing-fixtures-v1", threshold=0.9,
                  models={"small":"gemini-small", "large":"gemini-large", "embedding":"gemini-embedding"},
                  queries=[dict(id="offline", cache_outcome="miss", baseline=[baseline], optimized=[optimized])])
    result = comparison_metrics(ledger)
    assert result["optimized"]["estimated_cost_usd"] is None
    assert result["savings_usd"] is None
    assert "Token usage unavailable for gemini-small." in result["issues"]
