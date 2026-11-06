"""Offline end-to-end cache verification; only the SDK calls are mocked."""

from datetime import datetime, timedelta
from decimal import Decimal
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from google.genai import types

from embeddings import DIMENSIONS, MODEL, EmbeddingGenerator
from instrumentation import InstrumentedGenerator, JsonlRecorder, load_pricing
from semantic_cache import QueryEligibility, ResponseEligibility, SemanticCache


ANSWER_MODEL = "gemini-3.1-flash-lite"
ORIGINAL = "What is a Python tuple?"
SIMILAR = "Explain what a tuple is in Python."
DISSIMILAR = "How many sides does a triangle have?"
ANSWER = "A Python tuple is an immutable sequence of items."

# These fixed educational fixtures have been reviewed explicitly. This is not
# an automatic policy for approving arbitrary standalone-looking requests.
QUERY_REVIEWS = {
    query: QueryEligibility(
        query=query,
        context_independent=True,
        non_sensitive=True,
        time_independent=True,
        external_state_independent=True,
    )
    for query in (ORIGINAL, SIMILAR, DISSIMILAR)
}
ANSWER_REVIEW = ResponseEligibility(
    response=ANSWER, successful=True, not_safety_refusal=True, non_sensitive=True,
)


@pytest.mark.parametrize("embedding_tokens", [1000, 0, None], ids=[
    "reported-usage", "reported-zero", "unavailable-usage",
])
def test_answer_once_then_semantic_hit_after_reopening(tmp_path, embedding_tokens):
    pricing = load_pricing(Path(__file__).resolve().parents[1] / "model_pricing.json")
    database_path = tmp_path / "cache.sqlite3"
    metrics_path = tmp_path / "requests.jsonl"
    client = Mock()
    # Deliberately non-unit SDK vectors exercise the real normalization code.
    vectors = {
        ORIGINAL: [3.0, 0.0] + [0.0] * (DIMENSIONS - 2),
        SIMILAR: [9.6, 2.8] + [0.0] * (DIMENSIONS - 2),
        DISSIMILAR: [0.0, 4.0] + [0.0] * (DIMENSIONS - 2),
    }

    def embed_response(*, model, contents, config):
        assert model == MODEL
        assert config.task_type == "SEMANTIC_SIMILARITY"
        assert config.output_dimensionality == DIMENSIONS
        statistics = None if embedding_tokens is None else (
            types.ContentEmbeddingStatistics(token_count=embedding_tokens)
        )
        return types.EmbedContentResponse(embeddings=[
            types.ContentEmbedding(values=vectors[contents], statistics=statistics),
        ])

    client.models.embed_content.side_effect = embed_response
    # A second generation attempt fails immediately, even before count assertions.
    client.models.generate_content.side_effect = [
        types.GenerateContentResponse(
            candidates=[types.Candidate(
                content=types.Content(
                    role="model", parts=[types.Part.from_text(text=ANSWER)],
                ),
                finish_reason="STOP",
            )],
            usage_metadata=types.GenerateContentResponseUsageMetadata(
            cached_content_token_count=0, tool_use_prompt_token_count=0,
                prompt_token_count=100, candidates_token_count=20,
                thoughts_token_count=0, total_token_count=120,
            ),
        ),
    ]

    def make_generator():
        return InstrumentedGenerator(client, pricing, JsonlRecorder(metrics_path))

    def answer_query(cache, generator, query):
        """Test-only orchestration; production CLI remains independent."""
        review = QUERY_REVIEWS[query]
        result = cache.lookup(query, eligibility=review)
        if result.hit:
            return result.response, result.entry_id, result
        response = generator.generate_content(
            model=ANSWER_MODEL, request_kind="answer", contents=query,
        )
        assert response.candidates[0].finish_reason == types.FinishReason.STOP
        assert response.text == ANSWER_REVIEW.response
        entry_id = cache.save(
            query, response.text, ANSWER_MODEL,
            eligibility=review, response_eligibility=ANSWER_REVIEW,
        )
        return response.text, entry_id, result

    generator = make_generator()
    with SemanticCache(
        database_path, EmbeddingGenerator(generator), threshold=0.9,
    ) as cache:
        # Eligibility stays fail-closed, without spending any embedding tokens.
        denied = cache.lookup(ORIGINAL, eligibility=QueryEligibility(query=ORIGINAL))
        assert not denied.hit and denied.reason == "ineligible"
        assert cache.save(
            ORIGINAL, ANSWER, ANSWER_MODEL, eligibility=QUERY_REVIEWS[ORIGINAL],
        ) is None  # No explicit response review.
        client.models.embed_content.assert_not_called()
        client.models.generate_content.assert_not_called()

        answer, entry_id, initial_lookup = answer_query(cache, generator, ORIGINAL)
        assert not initial_lookup.hit and initial_lookup.reason == "miss"
        assert answer == ANSWER and entry_id is not None
        assert client.models.embed_content.call_count == 2
        client.models.generate_content.assert_called_once_with(
            model=ANSWER_MODEL, contents=ORIGINAL,
        )

    assert database_path.is_file()
    # Fresh cache, embedder, instrumentation wrapper, and recorder use the files
    # from the first instance; the real SQLite and similarity code are not mocked.
    reopened_generator = make_generator()
    with SemanticCache(
        database_path, EmbeddingGenerator(reopened_generator), threshold=0.9,
    ) as reopened:
        cached_answer, cached_id, hit = answer_query(
            reopened, reopened_generator, SIMILAR,
        )
        assert hit.hit and hit.reason == "hit"
        assert cached_answer == ANSWER and cached_id == entry_id
        assert hit.matched_query == ORIGINAL
        assert hit.similarity_score == pytest.approx(0.96)

        miss = reopened.lookup(DISSIMILAR, eligibility=QUERY_REVIEWS[DISSIMILAR])
        assert not miss.hit and miss.reason == "miss"
        assert miss.response is miss.entry_id is miss.matched_query is None
        assert miss.similarity_score is None
        assert client.models.generate_content.call_count == 1

    calls = client.models.embed_content.call_args_list
    assert [call.kwargs["contents"] for call in calls] == [
        ORIGINAL, ORIGINAL, SIMILAR, DISSIMILAR,
    ]
    records = [json.loads(line) for line in metrics_path.read_text(
        encoding="utf-8",
    ).splitlines()]
    assert [record["request_kind"] for record in records] == [
        "embedding", "answer", "embedding", "embedding", "embedding",
    ]
    for record in records:
        assert record["status"] == "success"
        assert record["latency_ms"] >= 0
        assert datetime.fromisoformat(record["timestamp_utc"]).utcoffset() == timedelta(0)
        assert record["pricing_version"] == pricing[record["model"]]["pricing_version"]

    embedding_records = [record for record in records if record["request_kind"] == "embedding"]
    assert len(embedding_records) == 4
    for record in embedding_records:
        assert record["model"] == MODEL
        assert record["input_tokens"] == embedding_tokens
        assert record["output_tokens"] is None
        if embedding_tokens is None:
            assert record["estimated_cost_usd"] is None
            assert record["cost_status"] == "usage_unavailable"
            assert record["usage_source"] is None
        else:
            expected = (
                Decimal(embedding_tokens) * Decimal(pricing[MODEL]["input_usd_per_million"])
                / Decimal(1_000_000)
            )
            assert Decimal(record["estimated_cost_usd"]) == expected
            assert record["cost_status"] == "estimated"
            assert record["usage_source"] == "embedding_statistics"

    answer_record = records[1]
    assert answer_record["model"] == ANSWER_MODEL
    assert answer_record["input_tokens"] == 100
    assert answer_record["output_tokens"] == answer_record["billed_output_tokens"] == 20
    expected_answer_cost = (
        100 * Decimal(pricing[ANSWER_MODEL]["input_usd_per_million"])
        + 20 * Decimal(pricing[ANSWER_MODEL]["output_usd_per_million"])
    ) / Decimal(1_000_000)
    assert Decimal(answer_record["estimated_cost_usd"]) == expected_answer_cost
    assert answer_record["cost_status"] == "estimated"
    # Instrumentation persists costs and timing, never query or response text.
    log_text = metrics_path.read_text(encoding="utf-8")
    assert all(text not in log_text for text in (ORIGINAL, SIMILAR, DISSIMILAR, ANSWER))
