"""Keyless, deterministic M3 report from committed predictions (drift-gated).

Selects the threshold on M3 validation, freezes it, applies to internal_test, and computes the
frontier / per-category / negation-slice / calibration / stability metrics — plus the PAIRED
comparison against the committed M2 NLI baseline on the identical internal_test pairs — from the
committed predictions alone (no model, no network). The disposition applies the pre-registered
uncertainty-aware bar (brief §10/§11). Model inference is the one author-run step; every number
here recomputes byte-for-byte in CI.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from polarity.baselines import select
from polarity.baselines.schema import BaselinePrediction, align_predictions_exact
from polarity.calibration import brier_score, expected_calibration_error
from polarity.calibration import fit as calfit
from polarity.eval import metrics
from polarity.eval.curves import partial_auc, recall_at_fpr_budgets
from polarity.train import collapse as collapse_mod
from polarity.train.config import SuccessCfg, TrainConfig, load_config
from polarity.types import Category, Label, Prediction

PRED_ROOT = Path("predictions/m3")
BASELINE_PRED = Path("predictions/m2/nli-cross-encoder")
REPORTS_ROOT = Path("reports/m3")
_GRID = tuple(round(0.05 * i, 2) for i in range(21))
_FPR_BUDGETS = (0.01, 0.005)


def _round(x: float) -> float:
    return round(x, 6)


def _load(path: Path) -> list[BaselinePrediction]:
    preds = [
        BaselinePrediction.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [p.pair_id for p in preds]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate pair_ids in {path}")
    return preds


def load(run_id: str, split: str, out_root: Path) -> list[BaselinePrediction]:
    return _load(out_root / PRED_ROOT / run_id / f"{split}.jsonl")


def load_baseline(split: str, out_root: Path) -> list[BaselinePrediction]:
    return _load(out_root / BASELINE_PRED / f"{split}.jsonl")


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


def _slice_stats(
    test: list[BaselinePrediction], category: Category, tau: float
) -> dict[str, object]:
    items = [p for p in test if p.category == category]
    positives = [p for p in items if p.gold == Label.CACHE_EQUIVALENT]
    negatives = [p for p in items if p.gold == Label.NOT_CACHE_EQUIVALENT]
    recall = sum(1 for p in positives if p.score >= tau) / len(positives) if positives else None
    fp_rate = sum(1 for p in negatives if p.score >= tau) / len(negatives) if negatives else None
    return {
        "n": len(items),
        "n_positive": len(positives),
        "n_negative": len(negatives),
        "recall_at_op": _round(recall) if recall is not None else None,
        "fp_rate_at_op": _round(fp_rate) if fp_rate is not None else None,
    }


def _calibration(
    val: list[BaselinePrediction], test: list[BaselinePrediction], cfg: TrainConfig
) -> dict[str, object]:
    v_scores = [p.score for p in val]
    v_labels = [1 if p.gold == Label.CACHE_EQUIVALENT else 0 for p in val]
    t_scores = [p.score for p in test]
    t_labels = [1 if p.gold == Label.CACHE_EQUIVALENT else 0 for p in test]

    candidates: list[calfit.Calibrator] = []
    for name in cfg.calibration.candidates:
        if name == "identity":
            candidates.append(calfit.identity())
        elif name == "temperature":
            candidates.append(calfit.fit_temperature(v_scores, v_labels))
        elif name == "platt":
            candidates.append(calfit.fit_platt(v_scores, v_labels))
        elif name == "isotonic":
            iso = calfit.fit_isotonic(v_scores, v_labels, min_obs=cfg.calibration.isotonic_min_obs)
            if iso is not None:
                candidates.append(iso)
    # Select the calibrator with the lowest VALIDATION NLL; apply it frozen to internal_test.
    best = min(candidates, key=lambda c: calfit.nll(c.apply(v_scores), v_labels))
    calibrated = best.apply(t_scores)
    per_method = {c.method: _round(calfit.nll(c.apply(v_scores), v_labels)) for c in candidates}
    return {
        "candidates_validation_nll": per_method,
        "selected_method": best.method,
        "selected_params": {
            name: _round(value) if isinstance(value, float) else value
            for name, value in best.params.items()
        },
        "ece_raw": _round(expected_calibration_error(t_scores, t_labels)),
        "ece_calibrated": _round(expected_calibration_error(calibrated, t_labels)),
        "brier_raw": _round(brier_score(t_scores, t_labels)),
        "brier_calibrated": _round(brier_score(calibrated, t_labels)),
    }


_FPR_BUDGET_GRID = (0.005, 0.01, 0.02, 0.05)


def _recall_on(positives: list[BaselinePrediction], tau: float) -> float:
    return sum(1 for p in positives if p.score >= tau) / len(positives) if positives else 0.0


def _paired_recall_bootstrap(
    m3_test: list[BaselinePrediction],
    nli_test: list[BaselinePrediction],
    tau_m3: float,
    tau_nli: float,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, object]:
    """Paired bootstrap of (recall_m3 - recall_nli) over the shared positive internal_test pairs."""
    aligned = [
        (m3_pred, nli_pred)
        for m3_pred, nli_pred in align_predictions_exact(
            m3_test, nli_test, left_name="m3", right_name="nli_baseline"
        )
        if m3_pred.gold == Label.CACHE_EQUIVALENT
    ]
    n = len(aligned)
    point = _recall_on([m for m, _ in aligned], tau_m3) - _recall_on(
        [b for _, b in aligned], tau_nli
    )
    if n == 0:
        return {
            "n_positive": 0,
            "delta_recall": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "n_boot": n_boot,
        }
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        m_hits = sum(1 for i in idx if aligned[i][0].score >= tau_m3)
        b_hits = sum(1 for i in idx if aligned[i][1].score >= tau_nli)
        deltas.append((m_hits - b_hits) / n)
    deltas.sort()
    lo = deltas[int(0.025 * n_boot)]
    hi = deltas[min(int(0.975 * n_boot), n_boot - 1)]
    return {
        "n_positive": n,
        "delta_recall": _round(point),
        "ci_low": _round(lo),
        "ci_high": _round(hi),
        "n_boot": n_boot,
    }


def _mcnemar(
    a: list[BaselinePrediction], b: list[BaselinePrediction], tau_a: float, tau_b: float
) -> dict[str, object]:
    a_only = b_only = 0
    for pa, pb in align_predictions_exact(a, b, left_name="m3", right_name="nli_baseline"):
        a_ok = (pa.score >= tau_a) == (pa.gold == Label.CACHE_EQUIVALENT)
        b_ok = (pb.score >= tau_b) == (pb.gold == Label.CACHE_EQUIVALENT)
        if a_ok and not b_ok:
            a_only += 1
        elif b_ok and not a_ok:
            b_only += 1
    denom = a_only + b_only
    stat = ((abs(a_only - b_only) - 1) ** 2) / denom if denom else 0.0
    return {
        "m3_correct_nli_wrong": a_only,
        "nli_correct_m3_wrong": b_only,
        "mcnemar_statistic_cc": _round(stat),
        "note": "continuity-corrected McNemar; stat > 3.84 ~ p < 0.05 (chi-square, 1 df)",
    }


def _disposition(
    success: SuccessCfg,
    m3_recall_ci: dict[str, Any],
    paired: dict[str, Any],
    m3_eqn: dict[str, Any],
    nli_eqn: dict[str, Any],
    m3_expl: dict[str, Any],
    nli_expl: dict[str, Any],
) -> dict[str, object]:
    delta_low = float(paired["ci_low"])
    delta_high = float(paired["ci_high"])
    m3_recall_low = float(m3_recall_ci["ci_low"])

    primary_material = delta_low > 0.0 and m3_recall_low > success.nli_baseline_recall

    m3_eqn_recall = m3_eqn["recall_at_op"]
    secondary_material = (
        m3_eqn_recall is not None
        and float(m3_eqn_recall) >= success.nli_baseline_eqn_recall + success.secondary_eqn_margin
    )

    m3_expl_fp = m3_expl["fp_rate_at_op"]
    nli_expl_fp = nli_expl["fp_rate_at_op"]
    guardrail_ok = (
        m3_expl_fp is None
        or nli_expl_fp is None
        or float(m3_expl_fp) <= float(nli_expl_fp) + success.guardrail_expl_fp_margin
    )

    if not guardrail_ok or delta_high < 0.0:
        suggested = "REGRESSION"
    elif primary_material and secondary_material:
        suggested = "STRONG_SUCCESS"
    elif primary_material or secondary_material:
        suggested = "PARTIAL_SUCCESS"
    else:
        suggested = "NO_MATERIAL_GAIN"

    return {
        "rule": success.rule,
        "primary_material": primary_material,
        "secondary_material": secondary_material,
        "guardrail_ok": guardrail_ok,
        "suggested_disposition": suggested,
        "note": (
            "signals only; the final disposition (incl. DATA_OR_OBJECTIVE_PROBLEM, a manual call) "
            "is stated with rationale in docs/M3-REPORT.md"
        ),
    }


def _training_health(
    cfg: TrainConfig,
    val_core: list[Prediction],
    test_core: list[Prediction],
    nli_test_core: list[Prediction],
) -> dict[str, Any]:
    """Keyless collapse diagnostic (brief §4) from committed predictions — the mechanics gate.

    A collapsed model (M3/M3.1) emits (near-)constant scores: no class separation, so no threshold
    discriminates and recall is 0 at every FP budget. Recomputed here for validation + internal_test
    and the NLI baseline (contrast), each with a verdict against the pre-registered thresholds.
    """

    def block(preds: list[Prediction]) -> dict[str, Any]:
        st = collapse_mod.stats_from_predictions(preds)
        return {**st.as_dict(), "collapse": collapse_mod.assess(st, cfg.collapse).as_dict()}

    return {
        "collapse_thresholds": cfg.collapse.model_dump(mode="json"),
        "validation": block(val_core),
        "internal_test": block(test_core),
        "nli_baseline_internal_test": block(nli_test_core),
    }


def _report(cfg: TrainConfig, out_root: Path) -> dict[str, Any]:
    run_id = cfg.experiment_id
    budget = cfg.selection.fpr_budget
    val = load(run_id, "validation", out_root)
    test = load(run_id, "internal_test", out_root)
    val_core = [p.core() for p in val]
    test_core = [p.core() for p in test]

    nli_val = load_baseline("validation", out_root)
    nli_test = load_baseline("internal_test", out_root)
    nli_val_core = [p.core() for p in nli_val]

    n_neg = sum(1 for p in test_core if p.gold == Label.NOT_CACHE_EQUIVALENT)
    n_pos = len(test_core) - n_neg

    tau_op = select.threshold_at_fpr(val_core, budget)
    tau_nli = select.threshold_at_fpr(nli_val_core, budget)
    tau0 = select.threshold_zero_fp(val_core)

    operating = {
        "val_selected_zero_fp": {
            "validation_tau": _round(tau0),
            "internal_test": _operating_point(test_core, tau0),
        }
    }
    for b in _FPR_BUDGETS:
        tau = select.threshold_at_fpr(val_core, b)
        operating[f"val_selected_fpr_le_{b}"] = {
            "validation_tau": _round(tau),
            "internal_test": _operating_point(test_core, tau),
        }

    m3_recall_ci = metrics.bootstrap_recall_ci(test_core, tau_op).model_dump(mode="json")
    m3_eqn = _slice_stats(test, Category.EQUIVALENT_NEGATION, tau_op)
    nli_eqn = _slice_stats(nli_test, Category.EQUIVALENT_NEGATION, tau_nli)
    m3_expl = _slice_stats(test, Category.EXPLICIT_NEGATION, tau_op)
    nli_expl = _slice_stats(nli_test, Category.EXPLICIT_NEGATION, tau_nli)

    paired = _paired_recall_bootstrap(
        test,
        nli_test,
        tau_op,
        tau_nli,
        n_boot=cfg.success.bootstrap_n,
        seed=cfg.success.bootstrap_seed,
    )

    return {
        "experiment_id": run_id,
        "n_validation": len(val),
        "n_internal_test": len(test),
        "n_test_positive": n_pos,
        "n_test_negative": n_neg,
        "fp_definition": metrics.FP_DEFINITION,
        "selection": {"metric": cfg.selection.metric, "fpr_budget": budget, "split": "validation"},
        "training_health": _training_health(cfg, val_core, test_core, [p.core() for p in nli_test]),
        "operating_points": operating,
        "primary_operating_point": {
            "selection": f"validation FPR<={budget}",
            "tau": _round(tau_op),
            "internal_test": _operating_point(test_core, tau_op),
        },
        "internal_test_curve": [
            r.model_dump(mode="json") for r in metrics.sweep(test_core, list(_GRID))
        ],
        "partial_auc_fpr": {
            "fpr_max": cfg.selection.pauc_fpr_max,
            "m3": partial_auc(test_core, cfg.selection.pauc_fpr_max),
            "nli_baseline": partial_auc([p.core() for p in nli_test], cfg.selection.pauc_fpr_max),
        },
        "recall_at_fpr_budgets": {
            "budgets": list(_FPR_BUDGET_GRID),
            "m3": recall_at_fpr_budgets(test_core, _FPR_BUDGET_GRID),
            "nli_baseline": recall_at_fpr_budgets([p.core() for p in nli_test], _FPR_BUDGET_GRID),
        },
        "recall_ci_at_operating": m3_recall_ci,
        "per_category_at_operating": [
            c.model_dump(mode="json") for c in metrics.per_category(test_core, tau_op)
        ],
        "equivalent_negation": {"m3": m3_eqn, "nli_baseline": nli_eqn},
        "explicit_negation": {"m3": m3_expl, "nli_baseline": nli_expl},
        "calibration": _calibration(val, test, cfg),
        "paired_vs_nli_baseline": {
            "selection": (
                f"each model's threshold selected on validation to target FPR<={budget * 100:g}%"
            ),
            "m3_tau": _round(tau_op),
            "nli_tau": _round(tau_nli),
            "recall_delta_bootstrap": paired,
            "mcnemar": _mcnemar(test, nli_test, tau_op, tau_nli),
            "nli_baseline_recall_reference_at_validation_selected_threshold": (
                cfg.success.nli_baseline_recall
            ),
            "nli_baseline_internal_test_at_selected_threshold": _operating_point(
                [p.core() for p in nli_test], tau_nli
            ),
        },
        "disposition": _disposition(
            cfg.success, m3_recall_ci, paired, m3_eqn, nli_eqn, m3_expl, nli_expl
        ),
    }


def _dump(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, indent=2) + "\n"


def _files(cfg: TrainConfig, out_root: Path) -> dict[str, str]:
    rep = _report(cfg, out_root)
    # Scope report filenames by experiment_id so multiple experiments (xp-003, xp-004, …) coexist
    # under reports/m3/ without clobbering each other.
    return {
        f"metrics-{cfg.experiment_id}.json": _dump(rep),
        f"disposition-{cfg.experiment_id}.json": _dump(rep["disposition"]),
    }


def build_all(
    cfg_path: Path = Path("configs/train/m3-deberta-v3-base.yaml"), out_root: Path = Path(".")
) -> dict[str, str]:
    cfg = load_config(out_root / cfg_path)
    files = _files(cfg, out_root)
    for fname, text in files.items():
        path = out_root / REPORTS_ROOT / fname
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return files


def check_all(
    cfg_path: Path = Path("configs/train/m3-deberta-v3-base.yaml"), out_root: Path = Path(".")
) -> list[str]:
    cfg = load_config(out_root / cfg_path)
    files = _files(cfg, out_root)
    drift: list[str] = []
    for fname, text in files.items():
        path = out_root / REPORTS_ROOT / fname
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            drift.append(fname)
    return drift


def has_predictions(cfg: TrainConfig, out_root: Path = Path(".")) -> bool:
    return (out_root / PRED_ROOT / cfg.experiment_id / "internal_test.jsonl").exists()


__all__ = ["build_all", "check_all", "has_predictions", "load", "load_baseline"]
