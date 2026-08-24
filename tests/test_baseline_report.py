from pathlib import Path

import pytest

from polarity.baselines import report
from polarity.baselines.schema import BaselinePrediction
from polarity.types import Category, Label

_PAIRS = [
    ("p1", "CACHE_EQUIVALENT", "paraphrase"),
    ("p2", "CACHE_EQUIVALENT", "paraphrase"),
    ("p3", "CACHE_EQUIVALENT", "equivalent_negation"),
    ("n1", "NOT_CACHE_EQUIVALENT", "explicit_negation"),
    ("n2", "NOT_CACHE_EQUIVALENT", "explicit_negation"),
    ("n3", "NOT_CACHE_EQUIVALENT", "explicit_negation"),
]
_SCORES = {
    "exact": {"p1": 1.0, "p2": 0.0, "p3": 0.0, "n1": 0.0, "n2": 0.0, "n3": 0.0},
    "normalized": {"p1": 1.0, "p2": 1.0, "p3": 0.0, "n1": 0.0, "n2": 0.0, "n3": 0.0},
    "embedding-cosine": {"p1": 0.9, "p2": 0.8, "p3": 0.6, "n1": 0.3, "n2": 0.5, "n3": 0.85},
    "adaptive": {"p1": 0.88, "p2": 0.75, "p3": 0.65, "n1": 0.2, "n2": 0.45, "n3": 0.7},
    "nli-cross-encoder": {"p1": 0.95, "p2": 0.9, "p3": 0.7, "n1": 0.1, "n2": 0.2, "n3": 0.4},
}


def _write(root: Path, name: str, split: str, preds: list[BaselinePrediction]) -> None:
    path = root / "predictions" / "m2" / name / f"{split}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(p.model_dump_json() for p in preds) + "\n", encoding="utf-8")


def _fixture(root: Path) -> None:
    for name in _SCORES:
        preds = _preds(name)
        _write(root, name, "validation", preds)
        _write(root, name, "internal_test", preds)


def _preds(name: str) -> list[BaselinePrediction]:
    scores = _SCORES[name]
    return [
        BaselinePrediction(
            pair_id=pid,
            gold=Label(gold),
            score=scores[pid],
            category=Category(cat),
            source="cannot",
            has_neg_cue=(cat != "paraphrase"),
            token_overlap=0.5,
            len_diff=0,
        )
        for pid, gold, cat in _PAIRS
    ]


def test_build_all_and_check_all(tmp_path: Path):
    _fixture(tmp_path)
    files = report.build_all(out_root=tmp_path)
    assert "metrics-nli-cross-encoder.json" in files
    assert "comparison.json" in files
    assert "decision.json" in files
    assert report.check_all(out_root=tmp_path) == []  # reproduces byte-for-byte


def test_comparison_names_validation_selected_and_realized_test_metrics(tmp_path: Path):
    _fixture(tmp_path)
    files = report.build_all(out_root=tmp_path)
    import json

    comparison = json.loads(files["comparison.json"])
    nli = comparison["per_baseline"]["nli-cross-encoder"]
    assert comparison["operating_point"] == "threshold selected on validation to target FPR<=1%"
    assert nli["recall_at_validation_selected_target_fpr_le_1pct_threshold"] == 1.0
    assert nli["internal_test_fp_at_validation_selected_target_fpr_le_1pct_threshold"] == 0
    assert (
        nli["realized_internal_test_fpr_at_validation_selected_target_fpr_le_1pct_threshold"] == 0.0
    )
    assert "recall_at_fpr_le_1pct" not in nli
    assert "recall_at_0fp" not in nli


def _assert_mcnemar_rejects(
    a: list[BaselinePrediction], b: list[BaselinePrediction], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        report._mcnemar(a, b, 0.5, 0.5)


def test_mcnemar_accepts_exact_alignment():
    result = report._mcnemar(_preds("nli-cross-encoder"), _preds("embedding-cosine"), 0.5, 0.5)
    assert "mcnemar_statistic_cc" in result


def test_mcnemar_rejects_missing_ids():
    _assert_mcnemar_rejects(
        _preds("nli-cross-encoder"),
        _preds("embedding-cosine")[:-1],
        "prediction count mismatch",
    )


def test_mcnemar_rejects_label_mismatch():
    a = _preds("nli-cross-encoder")
    b = _preds("embedding-cosine")
    b[0] = b[0].model_copy(update={"gold": Label.NOT_CACHE_EQUIVALENT})
    _assert_mcnemar_rejects(a, b, "gold label mismatch for pair_id p1")


def test_mcnemar_rejects_category_mismatch():
    a = _preds("nli-cross-encoder")
    b = _preds("embedding-cosine")
    b[0] = b[0].model_copy(update={"category": Category.HARD_PARAPHRASE})
    _assert_mcnemar_rejects(a, b, "category mismatch for pair_id p1")


def test_mcnemar_rejects_source_mismatch():
    a = _preds("nli-cross-encoder")
    b = _preds("embedding-cosine")
    b[0] = b[0].model_copy(update={"source": "paws-wiki"})
    _assert_mcnemar_rejects(a, b, "source mismatch for pair_id p1")


def test_drift_detected(tmp_path: Path):
    _fixture(tmp_path)
    report.build_all(out_root=tmp_path)
    (tmp_path / "reports" / "m2" / "decision.json").write_text("{}\n", encoding="utf-8")
    assert "decision.json" in report.check_all(out_root=tmp_path)


def test_deterministic_rebuild(tmp_path: Path):
    _fixture(tmp_path)
    report.build_all(out_root=tmp_path)
    first = (tmp_path / "reports" / "m2" / "metrics-nli-cross-encoder.json").read_text()
    report.build_all(out_root=tmp_path)
    second = (tmp_path / "reports" / "m2" / "metrics-nli-cross-encoder.json").read_text()
    assert first == second


def test_duplicate_predictions_fail(tmp_path: Path):
    _fixture(tmp_path)
    path = tmp_path / "predictions" / "m2" / "nli-cross-encoder" / "internal_test.jsonl"
    dup = BaselinePrediction(
        pair_id="p1",
        gold=Label.CACHE_EQUIVALENT,
        score=0.5,
        category=Category.PARAPHRASE,
        source="cannot",
        has_neg_cue=False,
        token_overlap=0.5,
        len_diff=0,
    )
    path.write_text(path.read_text() + dup.model_dump_json() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate pair_ids"):
        report.build_all(out_root=tmp_path)


def test_zero_fp_operating_point_frozen_from_validation(tmp_path: Path):
    # nli negatives max out at 0.4 on validation -> 0-FP threshold ~0.4; test positives
    # (0.95/0.9/0.7) all clear it, negatives (<=0.4) do not => recall 1.0 at 0 FP.
    _fixture(tmp_path)
    report.build_all(out_root=tmp_path)
    import json

    metrics = json.loads(
        (tmp_path / "reports" / "m2" / "metrics-nli-cross-encoder.json").read_text()
    )
    op = metrics["operating_points"]["val_selected_zero_fp"]["internal_test"]
    assert op["fp"] == 0
    assert op["recall"] == 1.0
