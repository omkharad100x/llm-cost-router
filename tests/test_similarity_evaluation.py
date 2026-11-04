"""Offline evaluation tests with hand-checked confusion matrices and fixtures."""

from copy import deepcopy
import json

import numpy as np
import pytest

from evaluate_similarity import (
    DEFAULT_DATASET, calculate_metrics, evaluate_pairs,
    load_dataset, load_embeddings, main,
)


@pytest.fixture
def evaluation_data():
    pairs = load_dataset()
    source, embeddings = load_embeddings()
    return pairs, source, embeddings


def test_dataset_coverage_and_reviewed_annotations(evaluation_data):
    pairs, source, embeddings = evaluation_data
    assert len(pairs) == 22
    assert sum(pair["expected"] == "SAFE_MATCH" for pair in pairs) == 8
    categories = {
        "negation", "quantities", "dates_years", "entities_names",
        "operations", "conditions_constraints",
    }
    for category in categories:
        examples = [pair for pair in pairs if pair["category"] == category]
        assert len(examples) >= 2
        assert all(pair["expected"] == "MUST_NOT_MATCH" for pair in examples)
    assert all(pair["query_a"] != pair["query_b"] for pair in pairs)
    assert all(len(pair["explanation"]) >= 20 for pair in pairs)
    assert "not Gemini" in source
    assert set(embeddings) == {
        pair[field] for pair in pairs for field in ("query_a", "query_b")
    }
    for vector in embeddings.values():
        assert np.linalg.norm(vector) == pytest.approx(1)


def test_default_fixture_metrics_include_all_four_outcomes(evaluation_data):
    pairs, _, embeddings = evaluation_data
    report = evaluate_pairs(pairs, embeddings)
    assert report["threshold"] == 0.9
    assert report["pair_count"] == 22
    assert report["metrics"] == {
        "true_positives": 6, "false_positives": 7,
        "true_negatives": 7, "false_negatives": 2,
        "precision": 6 / 13, "recall": 6 / 8, "false_hit_rate": 7 / 14,
    }
    assert len(report["results"]) == 22
    false_hits = [
        row for row in report["results"]
        if row["matched"] and row["expected"] == "MUST_NOT_MATCH"
    ]
    assert len(false_hits) == 7
    assert all(row["similarity_score"] >= 0.9 for row in false_hits)


def test_metrics_have_independent_hand_calculated_oracle():
    rows = [
        {"expected": "SAFE_MATCH", "matched": True},
        {"expected": "SAFE_MATCH", "matched": False},
        {"expected": "SAFE_MATCH", "matched": False},
        {"expected": "MUST_NOT_MATCH", "matched": True},
        {"expected": "MUST_NOT_MATCH", "matched": False},
    ]
    assert calculate_metrics(rows) == {
        "true_positives": 1, "false_positives": 1,
        "true_negatives": 1, "false_negatives": 2,
        "precision": 0.5, "recall": 1 / 3, "false_hit_rate": 0.5,
    }


@pytest.mark.parametrize("rows,precision,recall,false_hit_rate", [
    ([], None, None, None),
    ([{"expected": "SAFE_MATCH", "matched": False}], None, 0.0, None),
    ([{"expected": "MUST_NOT_MATCH", "matched": False}], None, None, 0.0),
    ([{"expected": "SAFE_MATCH", "matched": True}], 1.0, 1.0, None),
])
def test_zero_denominators_are_unavailable(rows, precision, recall, false_hit_rate):
    metrics = calculate_metrics(rows)
    assert metrics["precision"] == precision
    assert metrics["recall"] == recall
    assert metrics["false_hit_rate"] == false_hit_rate


@pytest.mark.parametrize("threshold,tp,fp,tn,fn", [
    (-1, 8, 14, 0, 0), (1, 0, 0, 14, 8),
])
def test_threshold_endpoints(evaluation_data, threshold, tp, fp, tn, fn):
    pairs, _, embeddings = evaluation_data
    metrics = evaluate_pairs(pairs, embeddings, threshold=threshold)["metrics"]
    assert [metrics[key] for key in (
        "true_positives", "false_positives", "true_negatives", "false_negatives",
    )] == [tp, fp, tn, fn]


def test_adjacent_thresholds_use_real_similarity_and_do_not_mutate_data(evaluation_data):
    pairs, _, embeddings = evaluation_data
    pair = pairs[:1]
    original_pairs, original_vectors = deepcopy(pair), deepcopy(embeddings)
    score = evaluate_pairs(pair, embeddings, threshold=-1)["results"][0]["similarity_score"]
    for threshold, matched in [
        (np.nextafter(score, -np.inf), True), (score, True),
        (np.nextafter(score, np.inf), False),
    ]:
        report = evaluate_pairs(pair, embeddings, threshold=threshold)
        assert report["results"][0]["matched"] is matched
    assert pair == original_pairs and embeddings == original_vectors


@pytest.mark.parametrize("threshold", [None, True, "0.9", float("nan"), float("inf"), -1.1, 1.1])
def test_invalid_thresholds_are_not_defaulted(evaluation_data, threshold):
    pairs, _, embeddings = evaluation_data
    with pytest.raises(ValueError, match="threshold"):
        evaluate_pairs(pairs, embeddings, threshold=threshold)


@pytest.mark.parametrize("mutation", ["label", "missing", "duplicate", "schema", "explanation"])
def test_malformed_dataset_rejected(tmp_path, mutation):
    data = json.loads(DEFAULT_DATASET.read_text(encoding="utf-8"))
    if mutation == "label":
        data["pairs"][0]["expected"] = "MAYBE"
    elif mutation == "missing":
        del data["pairs"][0]["query_a"]
    elif mutation == "duplicate":
        data["pairs"].append(data["pairs"][0])
    elif mutation == "schema":
        data["schema_version"] = 2
    else:
        data["pairs"][0]["explanation"] = " "
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(path)


@pytest.mark.parametrize("bad_vector", [[0, 0], [2, 0], [1, 0, 0], [float("nan"), 0]])
def test_malformed_vectors_fail_without_skipping_pairs(evaluation_data, bad_vector):
    pairs, _, embeddings = evaluation_data
    embeddings[pairs[0]["query_b"]] = bad_vector
    with pytest.raises(ValueError, match="Vectors"):
        evaluate_pairs(pairs, embeddings)


def test_missing_embedding_fails_clearly(evaluation_data):
    pairs, _, embeddings = evaluation_data
    del embeddings[pairs[0]["query_b"]]
    with pytest.raises(ValueError, match="Missing embedding for pair safe_tuple"):
        evaluate_pairs(pairs, embeddings)


def test_cli_emits_json_and_honors_threshold(capsys):
    assert main(["--threshold", "1"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["threshold"] == 1
    assert report["metrics"]["precision"] is None
    assert "synthetic" in report["embedding_source"]
    assert all("explanation" in row for row in report["results"])


def test_cli_rejects_invalid_threshold(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--threshold", "nan"])
    assert error.value.code == 2
    assert "threshold" in capsys.readouterr().err


def test_invalid_metric_labels_are_rejected():
    with pytest.raises(ValueError):
        calculate_metrics([{"expected": "UNKNOWN", "matched": True}])
