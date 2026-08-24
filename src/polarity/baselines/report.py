"""Keyless, deterministic M2 report from committed predictions.

Selects thresholds on validation, freezes them, applies to internal_test, and computes every
metric/curve/calibration/shortcut/comparison from the committed predictions alone — no models, no
network. ``build_all`` writes ``reports/m2/*``; ``check_all`` recomputes and compares byte-for-byte
(the drift gate). Calibration uses grid-search temperature scaling (pure math) so the report is
version-independent and fully reproducible.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from polarity.baselines import select
from polarity.baselines.schema import BaselinePrediction, align_predictions_exact
from polarity.calibration import brier_score, expected_calibration_error
from polarity.eval import metrics
from polarity.types import Category, Label, Prediction

PRED_ROOT = Path("predictions/m2")
REPORTS_ROOT = Path("reports/m2")
BASELINES = ("exact", "normalized", "embedding-cosine", "adaptive", "nli-cross-encoder")
_GRID = tuple(round(0.05 * i, 2) for i in range(21))
_FPR_BUDGETS = (0.01, 0.005)
_EPS = 1e-9
_PRIMARY_SELECTION = "threshold selected on validation to target FPR<=1%"


def _round(x: float) -> float:
    return round(x, 6)


def load(name: str, split: str, out_root: Path) -> list[BaselinePrediction]:
    path = out_root / PRED_ROOT / name / f"{split}.jsonl"
    preds = [
        BaselinePrediction.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [p.pair_id for p in preds]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate pair_ids in predictions/m2/{name}/{split}.jsonl")
    return preds


def _operating_point(test: list[Prediction], tau: float) -> dict[str, object]:
    c = metrics.confusion_at(test, tau)
    recall = c.tp / (c.tp + c.fn) if (c.tp + c.fn) else 0.0
    fpr = c.fp / (c.fp + c.tn) if (c.fp + c.tn) else 0.0
    precision = c.tp / (c.tp + c.fp) if (c.tp + c.fp) else 1.0
    return {
        "tau": _round(tau),
        "tp": c.tp,
        "fp": c.fp,
        "tn": c.tn,
        "fn": c.fn,
        "recall": _round(recall),
        "fpr": _round(fpr),
        "precision": _round(precision),
    }


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _fit_temperature(scores: list[float], labels: list[int]) -> float:
    best_t, best_nll = 1.0, math.inf
    for step in range(2, 21):  # T in 0.5 .. 5.0
        temp = round(0.25 * step, 2)
        nll = 0.0
        for p, y in zip(scores, labels, strict=True):
            q = min(max(_sigmoid(_logit(p) / temp), _EPS), 1.0 - _EPS)
            nll -= y * math.log(q) + (1 - y) * math.log(1.0 - q)
        if nll < best_nll:
            best_t, best_nll = temp, nll
    return best_t


def _calibration(
    val: list[BaselinePrediction], test: list[BaselinePrediction]
) -> dict[str, object]:
    v_scores = [p.score for p in val]
    v_labels = [1 if p.gold == Label.CACHE_EQUIVALENT else 0 for p in val]
    temp = _fit_temperature(v_scores, v_labels)
    t_scores = [p.score for p in test]
    t_labels = [1 if p.gold == Label.CACHE_EQUIVALENT else 0 for p in test]
    calibrated = [_sigmoid(_logit(p) / temp) for p in t_scores]
    return {
        "temperature": temp,
        "ece_raw": _round(expected_calibration_error(t_scores, t_labels)),
        "ece_calibrated": _round(expected_calibration_error(calibrated, t_labels)),
        "brier_raw": _round(brier_score(t_scores, t_labels)),
        "brier_calibrated": _round(brier_score(calibrated, t_labels)),
    }


def _slice_stats(
    test: list[BaselinePrediction], category: Category, tau: float
) -> dict[str, object]:
    items = [p for p in test if p.category == category]
    n = len(items)
    positives = [p for p in items if p.gold == Label.CACHE_EQUIVALENT]
    negatives = [p for p in items if p.gold == Label.NOT_CACHE_EQUIVALENT]
    recall = sum(1 for p in positives if p.score >= tau) / len(positives) if positives else None
    fp_rate = sum(1 for p in negatives if p.score >= tau) / len(negatives) if negatives else None
    return {
        "n": n,
        "n_positive": len(positives),
        "n_negative": len(negatives),
        "recall_at_op": _round(recall) if recall is not None else None,
        "fp_rate_at_op": _round(fp_rate) if fp_rate is not None else None,
    }


def _shortcut(test: list[BaselinePrediction], tau: float) -> dict[str, object]:
    def err_rate(items: list[BaselinePrediction]) -> float | None:
        if not items:
            return None
        wrong = sum(1 for p in items if (p.score >= tau) != (p.gold == Label.CACHE_EQUIVALENT))
        return _round(wrong / len(items))

    by_cue = {
        "with_negation_cue": err_rate([p for p in test if p.has_neg_cue]),
        "without_negation_cue": err_rate([p for p in test if not p.has_neg_cue]),
    }
    sources = sorted({p.source for p in test})
    by_source = {s: err_rate([p for p in test if p.source == s]) for s in sources}
    high_overlap = err_rate([p for p in test if p.token_overlap >= 0.5])
    low_overlap = err_rate([p for p in test if p.token_overlap < 0.5])
    return {
        "error_by_negation_cue": by_cue,
        "error_by_source": by_source,
        "error_by_token_overlap": {
            "high_overlap>=0.5": high_overlap,
            "low_overlap<0.5": low_overlap,
        },
    }


def _baseline_report(name: str, out_root: Path) -> dict[str, Any]:
    val = load(name, "validation", out_root)
    test = load(name, "internal_test", out_root)
    val_core = [p.core() for p in val]
    test_core = [p.core() for p in test]

    n_neg = sum(1 for p in test_core if p.gold == Label.NOT_CACHE_EQUIVALENT)
    n_pos = len(test_core) - n_neg
    tau0 = select.threshold_zero_fp(val_core)

    operating = {
        "val_selected_zero_fp": {
            "validation_tau": _round(tau0),
            "internal_test": _operating_point(test_core, tau0),
        }
    }
    for budget in _FPR_BUDGETS:
        tau = select.threshold_at_fpr(val_core, budget)
        operating[f"val_selected_fpr_le_{budget}"] = {
            "validation_tau": _round(tau),
            "internal_test": _operating_point(test_core, tau),
        }

    # The validation zero-observed-FP threshold is so strict it rejects nearly everything
    # (uninformative per-category); headline slices use the validation-selected target-FPR point.
    tau_op = select.threshold_at_fpr(val_core, 0.01)

    return {
        "baseline": name,
        "n_validation": len(val),
        "n_internal_test": len(test),
        "n_test_positive": n_pos,
        "n_test_negative": n_neg,
        "fp_definition": metrics.FP_DEFINITION,
        "fpr_resolution_note": (
            f"{n_neg} internal_test negatives => FPR granularity 1/{n_neg} = "
            f"{_round(1 / n_neg) if n_neg else 0.0}; sub-granularity FPR points are not resolvable."
        ),
        "operating_points": operating,
        "primary_operating_point": {
            "selection": _PRIMARY_SELECTION,
            "tau": _round(tau_op),
            "internal_test": _operating_point(test_core, tau_op),
        },
        "internal_test_curve": [
            r.model_dump(mode="json") for r in metrics.sweep(test_core, list(_GRID))
        ],
        "recall_ci_at_operating": metrics.bootstrap_recall_ci(test_core, tau_op).model_dump(
            mode="json"
        ),
        "per_category_at_operating": [
            c.model_dump(mode="json") for c in metrics.per_category(test_core, tau_op)
        ],
        "equivalent_negation_at_operating": _slice_stats(
            test, Category.EQUIVALENT_NEGATION, tau_op
        ),
        "explicit_negation_at_operating": _slice_stats(test, Category.EXPLICIT_NEGATION, tau_op),
        "calibration": _calibration(val, test),
        "shortcut_at_operating": _shortcut(test, tau_op),
    }


def _mcnemar(
    a: list[BaselinePrediction], b: list[BaselinePrediction], tau_a: float, tau_b: float
) -> dict[str, object]:
    b_only = c_only = 0
    for pa, pb in align_predictions_exact(
        a, b, left_name="first baseline", right_name="second baseline"
    ):
        a_ok = (pa.score >= tau_a) == (pa.gold == Label.CACHE_EQUIVALENT)
        b_ok = (pb.score >= tau_b) == (pb.gold == Label.CACHE_EQUIVALENT)
        if a_ok and not b_ok:
            b_only += 1
        elif b_ok and not a_ok:
            c_only += 1
    denom = b_only + c_only
    stat = ((abs(b_only - c_only) - 1) ** 2) / denom if denom else 0.0
    return {
        "a_correct_b_wrong": b_only,
        "b_correct_a_wrong": c_only,
        "mcnemar_statistic_cc": _round(stat),
        "note": "continuity-corrected McNemar; stat > 3.84 ~ p < 0.05 (chi-square, 1 df)",
    }


def _comparison(reports: dict[str, dict[str, Any]], out_root: Path) -> dict[str, Any]:
    rows = {}
    for name, rep in reports.items():
        op0 = rep["operating_points"]["val_selected_zero_fp"]["internal_test"]
        op1 = rep["primary_operating_point"]["internal_test"]
        eqn = rep["equivalent_negation_at_operating"]
        rows[name] = {
            "recall_at_validation_zero_observed_fp_threshold": op0["recall"],
            "internal_test_fp_at_validation_zero_observed_fp_threshold": op0["fp"],
            "realized_internal_test_fpr_at_validation_zero_observed_fp_threshold": op0["fpr"],
            "recall_at_validation_selected_target_fpr_le_1pct_threshold": op1["recall"],
            "precision_at_validation_selected_target_fpr_le_1pct_threshold": op1["precision"],
            "internal_test_fp_at_validation_selected_target_fpr_le_1pct_threshold": op1["fp"],
            "realized_internal_test_fpr_at_validation_selected_target_fpr_le_1pct_threshold": op1[
                "fpr"
            ],
            "equivalent_negation_recall_at_operating": eqn["recall_at_op"],
        }
    result: dict[str, Any] = {"operating_point": _PRIMARY_SELECTION, "per_baseline": rows}
    if {"nli-cross-encoder", "embedding-cosine"} <= set(reports):
        nli_v = load("nli-cross-encoder", "validation", out_root)
        emb_v = load("embedding-cosine", "validation", out_root)
        nli_t = load("nli-cross-encoder", "internal_test", out_root)
        emb_t = load("embedding-cosine", "internal_test", out_root)
        tau_nli = select.threshold_at_fpr([p.core() for p in nli_v], 0.01)
        tau_emb = select.threshold_at_fpr([p.core() for p in emb_v], 0.01)
        result["mcnemar_nli_vs_embedding_at_operating"] = _mcnemar(nli_t, emb_t, tau_nli, tau_emb)
    return result


def _decision(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    def recall_op(name: str) -> float:
        return float(reports[name]["primary_operating_point"]["internal_test"]["recall"])

    def eqn_recall(name: str) -> float | None:
        v = reports[name]["equivalent_negation_at_operating"]["recall_at_op"]
        return float(v) if v is not None else None

    threshold_best = max(
        recall_op("exact"),
        recall_op("normalized"),
        recall_op("embedding-cosine"),
        recall_op("adaptive"),
    )
    nli = recall_op("nli-cross-encoder")
    nli_eqn = eqn_recall("nli-cross-encoder")
    # Signals only (at each validation-selected target-FPR<=1% threshold); realized internal-test
    # FPR is reported separately. The final disposition is stated in docs/M2-REPORT.md.
    if nli >= 0.80 and (nli_eqn is not None and nli_eqn >= 0.70):
        suggested = "OFF_THE_SHELF_BASELINE_SUFFICIENT"
    elif nli > threshold_best + 0.02 and (nli_eqn is None or nli_eqn < 0.30):
        suggested = "CUSTOM_TRAINING_JUSTIFIED"
    elif nli <= threshold_best + 0.01:
        suggested = "BASELINE_RESULTS_INCONCLUSIVE"
    else:
        suggested = "CUSTOM_TRAINING_JUSTIFIED"
    return {
        "operating_point": _PRIMARY_SELECTION,
        "best_threshold_family_recall": _round(threshold_best),
        "nli_recall": _round(nli),
        "nli_equivalent_negation_recall": None if nli_eqn is None else _round(nli_eqn),
        "suggested_disposition": suggested,
        "note": "signals only; the report states the final disposition with rationale",
    }


def build_all(out_root: Path = Path(".")) -> dict[str, str]:
    reports = {
        name: _baseline_report(name, out_root)
        for name in BASELINES
        if _has_predictions(name, out_root)
    }
    files = {f"metrics-{name}.json": _dump(rep) for name, rep in reports.items()}
    files["comparison.json"] = _dump(_comparison(reports, out_root))
    files["decision.json"] = _dump(_decision(reports))
    for fname, text in files.items():
        path = out_root / REPORTS_ROOT / fname
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return files


def check_all(out_root: Path = Path(".")) -> list[str]:
    reports = {
        name: _baseline_report(name, out_root)
        for name in BASELINES
        if _has_predictions(name, out_root)
    }
    files = {f"metrics-{name}.json": _dump(rep) for name, rep in reports.items()}
    files["comparison.json"] = _dump(_comparison(reports, out_root))
    files["decision.json"] = _dump(_decision(reports))
    drift: list[str] = []
    for fname, text in files.items():
        path = out_root / REPORTS_ROOT / fname
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            drift.append(fname)
    return drift


def _has_predictions(name: str, out_root: Path) -> bool:
    return (out_root / PRED_ROOT / name / "internal_test.jsonl").exists()


def _dump(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, indent=2) + "\n"


__all__ = ["BASELINES", "build_all", "check_all", "load"]
