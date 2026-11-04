"""Offline component benchmarks. No real SDK client or environment loading."""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from decimal import Decimal
import hashlib
import json
import math
import platform
import sqlite3
from importlib.metadata import version
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace

from google.genai import types

from cache_config import DEFAULT_CONFIG, validate_threshold
from cache_schema import SCHEMA_VERSION
from embeddings import DIMENSIONS, MODEL, EmbeddingGenerator
from escalation import EscalationPipeline
from evaluate_similarity import (
    DEFAULT_DATASET, DEFAULT_EMBEDDINGS, calculate_metrics, evaluate_pairs,
    load_dataset, load_embeddings,
)
from evaluation_config import DEFAULT_EVALUATION_CONFIG, EvaluationConfig
from gemini_models import GeminiModels
from instrumentation import InstrumentedGenerator, load_pricing
from model_config import ModelConfig
from model_router import DEFAULT_ROUTING_CONFIG, ModelRouter, RoutingConfig
from semantic_cache import QueryEligibility, ResponseEligibility, SemanticCache


ROOT = Path(__file__).resolve().parent
DEFAULT_FIXTURES = ROOT / "evaluation" / "benchmark_fixtures.json"
DEFAULT_PRICING = ROOT / "model_pricing.json"
QUERY_FLAGS = {"context_independent", "non_sensitive", "time_independent", "external_state_independent"}
RESPONSE_FLAGS = {"successful", "not_safety_refusal", "non_sensitive"}
USAGE_FIELDS = {"prompt_token_count", "candidates_token_count", "thoughts_token_count", "total_token_count"}


@dataclass(frozen=True)
class BenchmarkConfig:
    thresholds: tuple[float, ...] = (0.8, DEFAULT_CONFIG.similarity_threshold, 0.95)
    lookup_repeats: int = 2
    models: ModelConfig = ModelConfig()
    routing: RoutingConfig = DEFAULT_ROUTING_CONFIG
    evaluation: EvaluationConfig = DEFAULT_EVALUATION_CONFIG

    def __post_init__(self):
        if not isinstance(self.thresholds, tuple) or not self.thresholds:
            raise ValueError("thresholds must be a nonempty tuple.")
        for threshold in self.thresholds:
            validate_threshold(threshold)
        if len(set(self.thresholds)) != len(self.thresholds):
            raise ValueError("Thresholds must be distinct.")
        if type(self.lookup_repeats) is not int or self.lookup_repeats < 1:
            raise ValueError("lookup_repeats must be a positive integer.")
        for value, kind in ((self.models, ModelConfig), (self.routing, RoutingConfig),
                            (self.evaluation, EvaluationConfig)):
            if not isinstance(value, kind):
                raise ValueError("Invalid benchmark component configuration.")
        if self.models.small_model == self.models.large_model:
            raise ValueError("Benchmark role fixtures require distinct model IDs.")


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def cache_metrics(rows):
    """Reuse existing confusion counts; undefined denominators remain None."""
    metrics = calculate_metrics(rows)
    hits = metrics["true_positives"] + metrics["false_positives"]
    misses = metrics["true_negatives"] + metrics["false_negatives"]
    return {
        **metrics, "true_hits": metrics["true_positives"],
        "false_hits": metrics["false_positives"], "true_misses": metrics["true_negatives"],
        "false_misses": metrics["false_negatives"],
        "hit_rate": ratio(hits, hits + misses), "miss_rate": ratio(misses, hits + misses),
        "false_miss_rate": ratio(metrics["false_negatives"],
                                 metrics["true_positives"] + metrics["false_negatives"]),
    }


def _request_totals(records, query_count):
    def total(field):
        values = [record.get(field) for record in records]
        return sum(values) if all(value is not None for value in values) else None

    known_costs = [Decimal(record["estimated_cost_usd"]) for record in records
                   if record.get("estimated_cost_usd") is not None]
    unknown = len(records) - len(known_costs)
    subtotal = sum(known_costs, Decimal(0))
    total_cost = None if unknown else subtotal
    statuses = Counter(record.get("cost_status", "metrics_unavailable") for record in records)
    status = "estimated" if records else "no_requests"
    if unknown:
        status = ("pricing_unavailable" if any(
            name in statuses for name in ("pricing_unavailable", "cached_pricing_unavailable")
        ) else "incomplete")
    latency = total("latency_ms")
    return {
        "requests": len(records),
        "cost": {
            "estimated_cost_usd": str(total_cost) if total_cost is not None else None,
            "average_cost_per_query_usd": str(total_cost / query_count)
            if total_cost is not None and query_count else None,
            "known_cost_subtotal_usd": str(subtotal), "unknown_cost_requests": unknown,
            "complete": unknown == 0, "status": status, "attempt_statuses": dict(sorted(statuses.items())),
        },
        "usage": {field: total(field) for field in
                  ("input_tokens", "output_tokens", "thinking_tokens", "billed_output_tokens")},
        "latency": {
            "total_ms": latency,
            "average_per_request_ms": ratio(latency, len(records)) if latency is not None else None,
            "average_per_query_ms": ratio(latency, query_count) if latency is not None else None,
            "source": "measured local mock SDK call overhead; not provider latency",
        },
    }


def summarize_requests(records, *, query_count):
    """Aggregate instrumentation records without inventing tokens or repricing."""
    if type(query_count) is not int or query_count < 0:
        raise ValueError("query_count must be a nonnegative integer.")
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("model"), str):
            raise ValueError("Expected instrumentation records with model names.")
        for field in ("input_tokens", "output_tokens", "thinking_tokens", "billed_output_tokens"):
            value = record.get(field)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("Token metrics must be nonnegative integers or None.")
        value = record.get("latency_ms")
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            raise ValueError("Latency must be finite and nonnegative or None.")
        value = record.get("estimated_cost_usd")
        if value is not None:
            try:
                amount = Decimal(value)
                if not amount.is_finite() or amount < 0:
                    raise ValueError
            except (ValueError, TypeError, ArithmeticError):
                raise ValueError("Cost must be a finite nonnegative decimal or None.") from None
    return {
        **_request_totals(records, query_count),
        "by_model": {model: _request_totals([r for r in records if r["model"] == model], query_count)
                     for model in sorted({record["model"] for record in records})},
    }


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _usage(value):
    if value is not None and (
        not isinstance(value, dict) or set(value) != USAGE_FIELDS
        or any(item is not None and (type(item) is not int or item < 0) for item in value.values())
    ):
        raise ValueError("Usage fixtures require explicit nonnegative token counts or null.")


def load_fixtures(path, pairs):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict) or type(data.get("schema_version")) is not int
        or data["schema_version"] != 1 or not _text(data.get("source"))
        or not isinstance(data.get("cache_reviews"), dict)
        or not isinstance(data.get("routing_cases"), list)
        or not _text(data.get("cache_miss_text"))
    ):
        raise ValueError("Expected version 1 benchmark fixtures.")
    if set(data["cache_reviews"]) != {pair["id"] for pair in pairs}:
        raise ValueError("Cache reviews must cover exactly the supplied pair IDs.")
    for pair in pairs:
        review = data["cache_reviews"][pair["id"]]
        if not isinstance(review, dict) or any(review.get(key) != pair[key] for key in ("query_a", "query_b")):
            raise ValueError("Cache review must be bound to the exact pair texts.")
        for key, fields in (("query_flags", QUERY_FLAGS), ("response_flags", RESPONSE_FLAGS)):
            flags = review.get(key)
            if not isinstance(flags, dict) or set(flags) != fields or any(type(v) is not bool for v in flags.values()):
                raise ValueError("Eligibility fixtures require explicit boolean flags.")
        if not _text(review.get("response")) or not _text(review.get("reason")):
            raise ValueError("Cache review requires a response and review reason.")
    if "cache_generation_usage" not in data or "embedding_token_count" not in data:
        raise ValueError("Usage fixtures must be explicit, including unavailable values.")
    _usage(data["cache_generation_usage"])
    count = data["embedding_token_count"]
    if count is not None and (type(count) is not int or count < 0):
        raise ValueError("Embedding usage must be a nonnegative integer or null.")
    seen = set()
    for case in data["routing_cases"]:
        if (
            not isinstance(case, dict) or not _text(case.get("id"))
            or not _text(case.get("query")) or not any(c.isalnum() for c in case["query"])
            or case.get("expected_classification") not in ("simple", "complex")
            or not isinstance(case.get("responses"), dict) or set(case["responses"]) != {"small", "large"}
        ):
            raise ValueError("Invalid routing case.")
        if case["id"] in seen:
            raise ValueError("Duplicate routing case ID.")
        seen.add(case["id"])
        for reply in case["responses"].values():
            if (
                not isinstance(reply, dict) or not _text(reply.get("text"))
                or reply.get("expected_quality") not in ("pass", "fail") or "usage" not in reply
            ):
                raise ValueError("Invalid model response fixture.")
            _usage(reply["usage"])
    return data


class _FixtureModels:
    """SDK-shaped deterministic adapter; no network-capable client is constructed."""

    def __init__(self, replies, vectors, embedding_tokens):
        self.replies = replies
        self.vectors = vectors
        self.embedding_tokens = embedding_tokens
        self.generation_calls = 0

    def generate_content(self, *, model, contents, **kwargs):
        reply = self.replies[(model, contents)]
        self.generation_calls += 1
        return types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(
                role="model", parts=[types.Part.from_text(text=reply["text"])],
            ))],
            usage_metadata=types.GenerateContentResponseUsageMetadata(**reply["usage"])
            if reply["usage"] is not None else None,
        )

    def embed_content(self, *, model, contents, config):
        if model != MODEL or config.output_dimensionality != DIMENSIONS:
            raise ValueError("Unexpected embedding configuration.")
        vector = self.vectors[contents]
        # Existing fixtures are 2D; zero padding preserves their cosine scores.
        values = list(vector) + [0.0] * (DIMENSIONS - len(vector))
        statistics = None if self.embedding_tokens is None else types.ContentEmbeddingStatistics(
            token_count=self.embedding_tokens,
        )
        return types.EmbedContentResponse(embeddings=[
            types.ContentEmbedding(values=values, statistics=statistics),
        ])


def _components(adapter, pricing, config, records):
    generator = InstrumentedGenerator(SimpleNamespace(models=adapter), pricing, records.append)
    return GeminiModels(generator, config=config.models), EmbeddingGenerator(generator)


def _query_review(query, review):
    return QueryEligibility(query=query, **review["query_flags"])


def benchmark_cache(pairs, vectors, fixtures, pricing, config, database):
    experiments = []
    for index, threshold in enumerate(config.thresholds):
        records, setup_records, workload_records, rows, latencies = [], [], [], [], []
        model = config.models.small_model
        avoided = correct_reuse = repeated_hits = generation_calls = 0
        for pair in pairs:
            review = fixtures["cache_reviews"][pair["id"]]
            # Keep responses isolated too: different pairs may share query text.
            replies = {
                (model, pair["query_b"]): {
                    "text": fixtures["cache_miss_text"], "usage": fixtures["cache_generation_usage"],
                },
                (model, pair["query_a"]): {
                    "text": review["response"], "usage": fixtures["cache_generation_usage"],
                },
            }
            adapter = _FixtureModels(replies, vectors, fixtures["embedding_token_count"])
            models, embedder = _components(adapter, pricing, config, records)
            namespace = f"threshold-{index}-pair-{pair['id']}"
            options = dict(threshold=threshold, namespace=namespace, clock=lambda: 1000.0)
            start = len(records)
            with SemanticCache(database, embedder, **options) as cache:
                eligible = _query_review(pair["query_a"], review).allows(pair["query_a"])
                response_review = ResponseEligibility(response=review["response"], **review["response_flags"])
                seed_id = None
                if eligible and response_review.allows(review["response"]):
                    answer = models.generate(pair["query_a"], role="small")
                    seed_id = cache.save(
                        pair["query_a"], answer.text, answer.model,
                        eligibility=_query_review(pair["query_a"], review),
                        response_eligibility=response_review,
                    )
            setup_records.extend(records[start:])
            start = len(records)
            first_hit = None
            # Reopen real SQLite before lookups; IDs are checked but not exported.
            with SemanticCache(database, embedder, **options) as cache:
                for repeat in range(config.lookup_repeats):
                    before_calls = adapter.generation_calls
                    started = time.perf_counter()
                    lookup = cache.lookup(pair["query_b"], eligibility=_query_review(pair["query_b"], review))
                    latencies.append((time.perf_counter() - started) * 1000)
                    if lookup.hit:
                        if lookup.entry_id != seed_id or lookup.response != review["response"]:
                            raise RuntimeError("Cache returned an unexpected fixture entry.")
                        avoided += 1
                        correct_reuse += pair["expected"] == "SAFE_MATCH"
                        repeated_hits += repeat > 0
                    else:
                        models.generate(pair["query_b"], role="small")
                    generation_calls += adapter.generation_calls - before_calls
                    if repeat == 0:
                        first_hit = lookup.hit
                        rows.append({
                            "id": pair["id"], "expected": pair["expected"], "matched": lookup.hit,
                            "eligible": eligible, "seeded": seed_id is not None,
                            "lookup_reason": lookup.reason, "similarity_score": lookup.similarity_score,
                        })
                    elif lookup.hit != first_hit:
                        raise RuntimeError("Repeated fixture lookups changed their decision.")
            workload_records.extend(records[start:])
        query_count = len(pairs) * config.lookup_repeats
        generation = [r for r in workload_records if r["request_kind"] != "embedding"]
        embeddings = [r for r in workload_records if r["request_kind"] == "embedding"]
        experiments.append({
            "threshold": threshold, "pair_count": len(pairs),
            "metrics": cache_metrics(rows),
            "eligible_metrics": cache_metrics([r for r in rows if r["eligible"]]),
            "eligible_pairs": sum(r["eligible"] for r in rows),
            "rows": rows,
            "reuse": {
                "lookup_count": query_count, "generation_calls": generation_calls,
                "generation_calls_avoided": avoided, "safe_reuse_count": correct_reuse,
                "unsafe_reuse_count": avoided - correct_reuse,
                "repeat_hits_after_first_lookup": repeated_hits,
            },
            "latency": {"total_lookup_ms": sum(latencies),
                        "average_lookup_ms": ratio(sum(latencies), len(latencies)),
                        "source": "measured SQLite + mocked embedding + similarity; generation excluded"},
            "setup_metrics": summarize_requests(setup_records, query_count=len(pairs)),
            "request_metrics": summarize_requests(workload_records, query_count=query_count),
            "generation_metrics": summarize_requests(generation, query_count=query_count),
            "embedding_metrics": summarize_requests(embeddings, query_count=query_count),
        })
    return experiments


def benchmark_routing(fixtures, pricing, config):
    records, cases = [], []
    for case in fixtures["routing_cases"]:
        replies = {(getattr(config.models, role + "_model"), case["query"]): reply
                   for role, reply in case["responses"].items()}
        adapter = _FixtureModels(replies, {}, None)
        models, _ = _components(adapter, pricing, config, records)
        result = EscalationPipeline(
            ModelRouter(models, config=config.routing), evaluation_config=config.evaluation,
        ).generate(case["query"])
        if adapter.generation_calls != result.total_attempts:
            raise RuntimeError("Model call count differs from recorded attempts.")
        cases.append({
            "id": case["id"], "classification": asdict(result.initial_routing.classification),
            "expected_classification": case["expected_classification"],
            "initial_role": result.initial_response.role,
            "routing_decision": asdict(result.initial_routing.decision),
            "escalated": result.escalated, "escalation_reason": result.escalation_reason,
            "final_attempt_index": result.final_attempt_index,
            "final_decision": result.final_evaluation.decision,
            "attempts": [{
                "role": attempt.role, "model": attempt.model,
                "quality_decision": attempt.evaluation.decision, "quality_score": attempt.evaluation.score,
                "quality_reasons": attempt.evaluation.reasons,
                "expected_quality": case["responses"][attempt.role]["expected_quality"],
            } for attempt in result.attempts],
        })
    count = len(cases)
    attempts = [attempt for case in cases for attempt in case["attempts"]]
    escalations = sum(case["escalated"] for case in cases)
    successes = sum(case["escalated"] and case["attempts"][1]["quality_decision"] == "pass" for case in cases)
    return {
        "cases": cases,
        "routing_metrics": {
            "query_count": count,
            "classifications": {label: sum(c["classification"]["label"] == label for c in cases)
                                for label in ("simple", "complex")},
            "initial_selections": {role: sum(c["initial_role"] == role for c in cases)
                                   for role in ("small", "large")},
            "routing_distribution": {role: ratio(sum(c["initial_role"] == role for c in cases), count)
                                     for role in ("small", "large")},
            "classification_fixture_agreement": ratio(sum(
                c["classification"]["label"] == c["expected_classification"] for c in cases), count),
            "all_attempt_roles": {role: sum(a["role"] == role for a in attempts) for role in ("small", "large")},
        },
        "escalation_metrics": {
            "initial_answers_accepted": sum(c["attempts"][0]["quality_decision"] == "pass" for c in cases),
            "escalations_triggered": escalations, "escalation_rate": ratio(escalations, count),
            "successful_escalations": successes, "escalation_success_rate": ratio(successes, escalations),
            "both_answers_fail": sum(c["escalated"] and all(a["quality_decision"] == "fail" for a in c["attempts"]) for c in cases),
            "total_model_attempts": len(attempts), "average_attempts_per_query": ratio(len(attempts), count),
        },
        "quality_metrics": {
            "attempt_passes": sum(a["quality_decision"] == "pass" for a in attempts),
            "attempt_failures": sum(a["quality_decision"] == "fail" for a in attempts),
            "final_passes": sum(c["final_decision"] == "pass" for c in cases),
            "average_attempt_score": ratio(sum(a["quality_score"] for a in attempts), len(attempts)),
            "fixture_label_agreement": ratio(sum(a["quality_decision"] == a["expected_quality"] for a in attempts), len(attempts)),
            "factual_correctness_verified": False,
        },
        "request_metrics": summarize_requests(records, query_count=count),
    }


def run_benchmarks(*, config=BenchmarkConfig(), dataset=DEFAULT_DATASET,
                   embeddings=DEFAULT_EMBEDDINGS, fixtures=DEFAULT_FIXTURES, pricing=DEFAULT_PRICING):
    if not isinstance(config, BenchmarkConfig):
        raise ValueError("config must be a BenchmarkConfig instance.")
    pairs = load_dataset(dataset)
    source, vectors = load_embeddings(embeddings)
    fixture_data = load_fixtures(fixtures, pairs)
    rates = load_pricing(pricing)
    # Validate vectors using the existing evaluator before creating temporary data.
    similarity = [evaluate_pairs(pairs, vectors, threshold=t) for t in config.thresholds]
    if any(len(vectors[pair[key]]) > DIMENSIONS for pair in pairs for key in ("query_a", "query_b")):
        raise ValueError("Embedding fixtures exceed cache dimensions.")
    with TemporaryDirectory(prefix="llm-router-benchmark-") as directory:
        cache = benchmark_cache(pairs, vectors, fixture_data, rates, config, Path(directory) / "cache.sqlite3")
    routing = benchmark_routing(fixture_data, rates, config)
    return {
        "report_schema_version": 1,
        "runtime": {
            "python": platform.python_version(), "platform": platform.system(),
            "sqlite": sqlite3.sqlite_version,
            "packages": {name: version(name) for name in ("google-genai", "numpy")},
        },
        "implementation_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "benchmarks.py", "semantic_cache.py", "cache_schema.py", "cache_config.py",
                "semantic_similarity.py", "embeddings.py", "evaluate_similarity.py",
                "model_router.py", "gemini_models.py", "model_config.py", "instrumentation.py",
                "query_complexity.py", "complexity_config.py", "answer_evaluator.py",
                "evaluation_config.py", "escalation.py",
            )
        },
        "configuration": {**asdict(config), "cache_schema_version": SCHEMA_VERSION,
                          "cache_version": DEFAULT_CONFIG.cache_version,
                          "ttl_seconds": 86400, "fixed_cache_clock": 1000.0, "pricing": rates},
        "datasets": {
            "pair_count": len(pairs), "routing_case_count": len(fixture_data["routing_cases"]),
            "embedding_source": source, "fixture_source": fixture_data["source"],
            "sha256": {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                       for name, path in (("pairs", dataset), ("embeddings", embeddings),
                                          ("fixtures", fixtures), ("pricing", pricing))},
        },
        "similarity_only": similarity,
        "cache_benchmarks": cache, "pipeline_benchmark": routing,
        "measurement_notes": [
            "Offline synthetic vectors and mock SDK text/token fixtures; no actual API spend or real Gemini quality measured.",
            "Generation timings measure instrumented mock SDK execution, not full routing/evaluation time or provider latency.",
            "Lookup timings include mock embedding, normalization, SQLite, and cosine search; generation is timed separately.",
            "One stored candidate per isolated pair namespace; zero padding to 768 dimensions; reopened SQLite; fixed TTL clock.",
            "All cache reviews are explicit and bound to exact fixture texts. Ineligible cases remain in overall cache counts.",
            "Warm seed generation/storage metrics are separate from repeated lookup/fallback workload metrics.",
            "Misses generate but are not saved, preserving the same candidate set across threshold comparisons.",
            "Avoided generation includes unsafe false hits; safe reuse is reported separately. No monetary savings baseline is claimed.",
            "Cost is estimated from the unchanged pricing file and synthetic usage; missing usage or prices make totals incomplete.",
            "Average cost per query divides by all scenario queries, including cache hits. By-model averages use the same denominator.",
            "Threshold experiments and the routing experiment are separate workloads and must not be added as one production total.",
            "Fixture label agreement is a regression check, not independent classification accuracy or factual verification.",
            "Empty workloads have zero requests/totals and undefined rates; unknown fields on actual requests remain null.",
            "Only timing fields vary between repeated runs; wall-clock timestamps and random cache IDs are not exported.",
        ],
    }


def format_summary(report):
    lines = ["OFFLINE SYNTHETIC BENCHMARK (not production performance)"]
    for experiment in report["cache_benchmarks"]:
        metrics = experiment["metrics"]
        lines.append(
            f"Cache threshold={experiment['threshold']}: "
            f"TP/FP/TN/FN={metrics['true_hits']}/{metrics['false_hits']}/"
            f"{metrics['true_misses']}/{metrics['false_misses']}; "
            f"hit rate={metrics['hit_rate']}; avoided calls={experiment['reuse']['generation_calls_avoided']}; "
            f"mean lookup ms={experiment['latency']['average_lookup_ms']}"
        )
    pipeline = report["pipeline_benchmark"]
    lines.append("Routing: " + json.dumps(pipeline["routing_metrics"], sort_keys=True))
    lines.append("Escalation: " + json.dumps(pipeline["escalation_metrics"], sort_keys=True))
    lines.append("Pipeline cost: " + json.dumps(pipeline["request_metrics"]["cost"], sort_keys=True))
    lines.append("Pipeline usage: " + json.dumps(pipeline["request_metrics"]["usage"], sort_keys=True))
    lines.append("Pipeline latency: " + json.dumps(pipeline["request_metrics"]["latency"], sort_keys=True))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.8, 0.9, 0.95])
    parser.add_argument("--lookup-repeats", type=int, default=2)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--pricing", type=Path, default=DEFAULT_PRICING)
    parser.add_argument("--output", type=Path, help="Write a new JSON file; existing files are never overwritten.")
    args = parser.parse_args(argv)
    try:
        config = BenchmarkConfig(thresholds=tuple(args.thresholds), lookup_repeats=args.lookup_repeats)
        report = run_benchmarks(config=config, dataset=args.dataset, embeddings=args.embeddings,
                                fixtures=args.fixtures, pricing=args.pricing)
        serialized = json.dumps(report, indent=2, allow_nan=False)
        if args.output:
            with args.output.open("x", encoding="utf-8") as file:
                file.write(serialized + "\n")
        else:
            print(serialized)
    except (OSError, ValueError):
        parser.error("Invalid benchmark inputs or report destination; check fixture schemas and use a new output path.")
    print(format_summary(report), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
