"""Evaluate labeled query pairs offline using precomputed vectors and cosine search.

No client, environment variables, cache database, or live API is used.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

from cache_config import DEFAULT_CONFIG, validate_threshold
from semantic_similarity import find_most_similar


DATA_DIRECTORY = Path(__file__).resolve().parent / "evaluation"
DEFAULT_DATASET = DATA_DIRECTORY / "query_pairs.json"
DEFAULT_EMBEDDINGS = DATA_DIRECTORY / "synthetic_embeddings.json"
LABELS = {"SAFE_MATCH", "MUST_NOT_MATCH"}


def load_dataset(path=DEFAULT_DATASET):
    """Load versioned annotations; malformed data must fail before evaluation."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict) or type(data.get("schema_version")) is not int
        or data["schema_version"] != 1 or not isinstance(data.get("pairs"), list)
    ):
        raise ValueError("Expected dataset schema_version 1 and a pairs list.")
    seen_ids, seen_pairs = set(), set()
    for pair in data["pairs"]:
        required = ("id", "category", "query_a", "query_b", "expected", "explanation")
        if not isinstance(pair, dict) or not all(
            isinstance(pair.get(field), str) and pair[field].strip()
            for field in required
        ):
            raise ValueError("Every pair needs nonempty text fields and an explanation.")
        if pair["expected"] not in LABELS:
            raise ValueError("Expected label SAFE_MATCH or MUST_NOT_MATCH.")
        identity = (pair["query_a"], pair["query_b"])
        if pair["id"] in seen_ids or identity in seen_pairs:
            raise ValueError("Duplicate pair ID or ordered query pair.")
        seen_ids.add(pair["id"])
        seen_pairs.add(identity)
    return data["pairs"]


def load_embeddings(path=DEFAULT_EMBEDDINGS):
    """Read text-to-vector fixtures; vector validation occurs during evaluation."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict) or type(data.get("schema_version")) is not int
        or data["schema_version"] != 1
        or not isinstance(data.get("source"), str) or not data["source"].strip()
        or not isinstance(data.get("embeddings"), dict)
    ):
        raise ValueError("Expected version 1 embeddings with a source and vector mapping.")
    return data["source"], data["embeddings"]


def calculate_metrics(results):
    """SAFE_MATCH is positive; undefined ratios are None, not zero."""
    counts = Counter()
    for result in results:
        if result.get("expected") not in LABELS or type(result.get("matched")) is not bool:
            raise ValueError("Metric results require a valid expected label and boolean matched.")
        positive = result["expected"] == "SAFE_MATCH"
        if positive:
            counts["true_positives" if result["matched"] else "false_negatives"] += 1
        else:
            counts["false_positives" if result["matched"] else "true_negatives"] += 1
    tp, fp, tn, fn = (counts[key] for key in (
        "true_positives", "false_positives", "true_negatives", "false_negatives",
    ))

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    return {
        "true_positives": tp, "false_positives": fp,
        "true_negatives": tn, "false_negatives": fn,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "false_hit_rate": ratio(fp, fp + tn),
    }


def evaluate_pairs(pairs, embeddings, *, threshold=DEFAULT_CONFIG.similarity_threshold):
    """Evaluate each pair independently; labels never control vector generation.

    embeddings maps query text to normalized, compatible precomputed vectors.
    This assesses similarity decisions only, not cache eligibility or answer quality.
    """
    threshold = validate_threshold(threshold)
    results = []
    for pair in pairs:
        if pair["expected"] not in LABELS:
            raise ValueError("Expected label SAFE_MATCH or MUST_NOT_MATCH.")
        try:
            query_vector = embeddings[pair["query_a"]]
            candidates = {pair["query_b"]: embeddings[pair["query_b"]]}
        except KeyError as error:
            raise ValueError(f"Missing embedding for pair {pair['id']}.") from error
        match = find_most_similar(query_vector, candidates, threshold=threshold)
        # A rejected candidate still needs its actual score in the diagnostic report.
        scored = match or find_most_similar(query_vector, candidates, threshold=-1)
        results.append({
            **pair, "similarity_score": scored.score, "matched": match is not None,
            "predicted": "SAFE_MATCH" if match is not None else "MUST_NOT_MATCH",
        })
    return {
        "threshold": float(threshold), "pair_count": len(results),
        "metrics": calculate_metrics(results), "results": results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIG.similarity_threshold)
    args = parser.parse_args(argv)
    try:
        validate_threshold(args.threshold)
        pairs = load_dataset(args.dataset)
        source, embeddings = load_embeddings(args.embeddings)
        report = evaluate_pairs(pairs, embeddings, threshold=args.threshold)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    report["embedding_source"] = source
    report["scope"] = "Pairwise similarity only; no cache eligibility or generation."
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
