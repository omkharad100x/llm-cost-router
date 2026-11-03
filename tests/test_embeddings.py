"""Embedding tests use mocked clients and SDK response objects, never live APIs."""

import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from google.genai import errors, types

from embeddings import DIMENSIONS, MODEL, EmbeddingGenerator, EmbeddingResponseError
from instrumentation import InstrumentedGenerator, embedding_usage_metrics, load_pricing


RATES = {"input_usd_per_million": "0.15", "output_usd_per_million": "0",
         "pricing_version": "test"}


def values(first=3, second=4):
    return [first, second] + [0] * (DIMENSIONS - 2)


def response(vector=None, token_count=None, truncated=None):
    return types.EmbedContentResponse(embeddings=[
        types.ContentEmbedding(
            values=values() if vector is None else vector,
            statistics=types.ContentEmbeddingStatistics(
                token_count=token_count, truncated=truncated
            ),
        )
    ])


def setup_embedder(sdk_response, pricing=None, recorder=None):
    client = Mock()
    client.models.embed_content.return_value = sdk_response
    records = []
    generator = InstrumentedGenerator(
        client, {MODEL: RATES} if pricing is None else pricing,
        records.append if recorder is None else recorder,
    )
    return EmbeddingGenerator(generator), client, records


def test_normalizes_and_forwards_existing_client_configuration():
    embedder, client, records = setup_embedder(response(token_count=1000))
    with patch("instrumentation.time.perf_counter", side_effect=[10, 10.125]):
        vector = embedder.embed_query("  Synthetic query  ")
    assert vector.shape == (DIMENSIONS,)
    assert vector.dtype == np.float64
    np.testing.assert_allclose(vector[:2], [0.6, 0.8])
    assert np.linalg.norm(vector) == pytest.approx(1)
    assert np.dot(vector, vector) == pytest.approx(1)
    args = client.models.embed_content.call_args.kwargs
    assert args["model"] == MODEL
    assert args["contents"] == "Synthetic query"
    assert args["config"].task_type == "SEMANTIC_SIMILARITY"
    assert args["config"].output_dimensionality == DIMENSIONS
    client.models.generate_content.assert_not_called()
    client.models.count_tokens.assert_not_called()
    assert len(records) == 1
    record = records[0]
    assert record["request_kind"] == "embedding"
    assert record["status"] == "success"
    assert record["model"] == MODEL
    assert record["latency_ms"] == 125
    assert record["timestamp_utc"].endswith("+00:00")
    assert record["input_tokens"] == 1000
    assert record["output_tokens"] is None
    assert Decimal(record["estimated_cost_usd"]) == Decimal("0.00015")
    assert "Synthetic query" not in json.dumps(record)
    assert "values" not in record


@pytest.mark.parametrize("text", ["", " \n\t", None, 42, [], b"query"])
def test_invalid_input_never_calls_sdk_or_records_request(text):
    embedder, client, records = setup_embedder(response())
    with pytest.raises(ValueError, match="nonempty text"):
        embedder.embed_query(text)
    client.models.embed_content.assert_not_called()
    assert records == []


@pytest.mark.parametrize("sdk_response", [
    None, SimpleNamespace(), types.EmbedContentResponse(),
    types.EmbedContentResponse(embeddings=[]),
    types.EmbedContentResponse(embeddings=[types.ContentEmbedding(), types.ContentEmbedding()]),
    types.EmbedContentResponse(embeddings=[types.ContentEmbedding()]),
    response([], truncated=False),
    response([1, 2]),
    response([0] * DIMENSIONS),
    response(values(float("nan"), 1)),
    response(values(float("inf"), 1)),
    response(values(float("-inf"), 1)),
    response(truncated=True),
    SimpleNamespace(embeddings=[SimpleNamespace(values=["1"] * DIMENSIONS)]),
    SimpleNamespace(embeddings=[SimpleNamespace(values=[True] * DIMENSIONS)]),
    SimpleNamespace(embeddings=[SimpleNamespace(values=[[1]] * DIMENSIONS)]),
    SimpleNamespace(embeddings=[SimpleNamespace(values=[1j] * DIMENSIONS)]),
])
def test_unexpected_response_rejected_but_sdk_request_recorded(sdk_response):
    embedder, client, records = setup_embedder(sdk_response)
    with pytest.raises(EmbeddingResponseError):
        embedder.embed_query("Synthetic")
    client.models.embed_content.assert_called_once()
    assert len(records) == 1
    # Status measures the SDK invocation; local validation happens afterward.
    assert records[0]["status"] == "success"


@pytest.mark.parametrize("magnitude", [1e308, 1e-308])
def test_stable_normalization_of_large_and_small_vectors(magnitude):
    embedder, _, _ = setup_embedder(response(values(magnitude, magnitude)))
    vector = embedder.embed_query("Synthetic")
    assert np.isfinite(vector).all()
    assert np.linalg.norm(vector) == pytest.approx(1)


@pytest.mark.parametrize("tokens,expected", [
    (None, None), (0, 0), (10, 10), (-1, None),
    (1.5, None), (float("nan"), None), (float("inf"), None),
])
def test_sdk_token_statistics_preserve_unknown_and_zero(tokens, expected):
    metrics = embedding_usage_metrics(response(token_count=tokens), RATES)
    assert metrics["input_tokens"] == expected
    if expected is None:
        assert metrics["estimated_cost_usd"] is None
        assert metrics["usage_source"] is None
    else:
        assert Decimal(metrics["estimated_cost_usd"]) == Decimal(expected) * Decimal("0.15") / 1000000
        assert metrics["usage_source"] == "embedding_statistics"


def test_no_statistics_characters_are_not_converted_to_tokens():
    sdk_response = types.EmbedContentResponse(
        embeddings=[types.ContentEmbedding(values=values())],
        metadata=types.EmbedContentMetadata(billable_character_count=100),
    )
    embedder, _, records = setup_embedder(sdk_response)
    embedder.embed_query("Synthetic")
    assert records[0]["input_tokens"] is None
    assert records[0]["estimated_cost_usd"] is None


def test_missing_pricing_retains_usage():
    embedder, _, records = setup_embedder(response(token_count=10), pricing={})
    embedder.embed_query("Synthetic")
    assert records[0]["input_tokens"] == 10
    assert records[0]["estimated_cost_usd"] is None
    assert records[0]["cost_status"] == "pricing_unavailable"


def test_api_error_preserved_and_recorded_without_sensitive_details():
    embedder, client, records = setup_embedder(None)
    error = errors.ServerError(503, {"error": {"message": "sensitive provider detail"}})
    client.models.embed_content.side_effect = error
    with pytest.raises(errors.ServerError) as caught:
        embedder.embed_query("private query")
    assert caught.value is error
    assert records[0]["status"] == "error"
    assert records[0]["latency_ms"] >= 0
    assert records[0]["input_tokens"] is None
    assert records[0]["estimated_cost_usd"] is None
    assert "private" not in json.dumps(records)
    assert "sensitive" not in json.dumps(records)


@pytest.mark.parametrize("fails", [False, True])
def test_recorder_failure_does_not_replace_result_or_api_error(fails, capsys):
    embedder, client, _ = setup_embedder(
        response(), recorder=Mock(side_effect=OSError("private file path"))
    )
    if fails:
        error = RuntimeError("provider")
        client.models.embed_content.side_effect = error
        with pytest.raises(RuntimeError) as caught:
            embedder.embed_query("Synthetic")
        assert caught.value is error
    else:
        assert np.linalg.norm(embedder.embed_query("Synthetic")) == pytest.approx(1)
    assert capsys.readouterr().err == "Warning: request metrics could not be recorded.\n"


def test_shared_pricing_file_contains_embedding_rate():
    pricing = load_pricing(Path(__file__).resolve().parents[1] / "model_pricing.json")
    assert pricing[MODEL]["input_usd_per_million"] == "0.15"
