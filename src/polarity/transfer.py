"""M6 — the sealed Veritas transfer arm. One-way, frozen, evaluated exactly once.

The threshold(s) and calibrator are derived from the M3.3 model's VALIDATION predictions and frozen
in a protocol BEFORE the Veritas pairs are ever scored (docs/M6-TRANSFER-PROTOCOL.md). Scoring the
75-pair sealed set (``golden.json``) is the only step that touches Veritas, and it happens once.

Positive class = CACHE_EQUIVALENT; a false positive is a NOT pair predicted CACHE_EQUIVALENT — the
failure that serves a wrong cached answer. Metrics are recomputed keylessly from the committed
per-pair scores; the transfer is drift-gated like every other Polarity report.
"""

from __future__ import annotations

import hashlib
import json
import math
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from polarity.artifact import ArtifactIdentity, ArtifactReleaseContext
from polarity.baselines import select
from polarity.baselines.schema import BaselinePrediction
from polarity.calibration import expected_calibration_error
from polarity.calibration import fit as calfit
from polarity.eval import metrics
from polarity.train.config import TrainConfig, recipe_fingerprint
from polarity.types import Label, Prediction

GOLDEN_PATH = Path("data/snapshots/veritas-golden/golden.json")
PRED_PATH = Path("predictions/m6/veritas")
REPORTS_ROOT = Path("reports/m6")
# Frozen operating points (validation-selected; identical protocol to M3.3). Reported exactly.
OP_BUDGETS = (0.0, 0.01, 0.05)  # 0.0 = the 0-observed-FP point
OP_KEYS = {"zero_fp", "fpr_le_0.01", "fpr_le_0.05"}
OP_SPECS = {
    "zero_fp": ("zero_observed_fp", None),
    "fpr_le_0.01": ("fpr_budget", 0.01),
    "fpr_le_0.05": ("fpr_budget", 0.05),
}


class ReproductionError(RuntimeError):
    """A candidate/committed comparison cannot produce a controlled verdict."""


class ScoringPreflightError(RuntimeError):
    """The requested scoring configuration does not match the frozen xp-006 identity."""


class GoldenScore(BaseModel):
    """One sealed-set pair scored once by the frozen model (the immutable transfer artifact)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    group: str
    pair_type: str
    gold_equivalent: bool
    raw_score: float  # P(CACHE_EQUIVALENT) from the frozen checkpoint
    calibrated: float  # frozen isotonic calibrator applied to raw_score


class SelectionMetrics(BaseModel):
    """Observed confusion counts and rates on the split that selected an operating point."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tp: int = Field(ge=0)
    fp: int = Field(ge=0)
    tn: int = Field(ge=0)
    fn: int = Field(ge=0)
    recall: float = Field(ge=0.0, le=1.0)
    fpr: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def rates_match_counts(self) -> SelectionMetrics:
        expected_recall = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        expected_fpr = self.fp / (self.fp + self.tn) if self.fp + self.tn else 0.0
        if not math.isclose(self.recall, expected_recall, abs_tol=5e-7):
            raise ValueError("selection recall does not match tp/fn counts")
        if not math.isclose(self.fpr, expected_fpr, abs_tol=5e-7):
            raise ValueError("selection FPR does not match fp/tn counts")
        return self


class FrozenOperatingPoint(BaseModel):
    """One validation-selected decision rule, including what validation actually observed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    selection_split: Literal["validation"]
    selection_rule: Literal["zero_observed_fp", "fpr_budget"]
    target_fpr: float | None = Field(default=None, ge=0.0, le=1.0)
    threshold: float
    observed_selection: SelectionMetrics

    @model_validator(mode="after")
    def rule_is_consistent(self) -> FrozenOperatingPoint:
        if not math.isfinite(self.threshold):
            raise ValueError("threshold must be finite")
        if self.selection_rule == "zero_observed_fp":
            if self.target_fpr is not None:
                raise ValueError("zero-observed-FP rule must not declare a target FPR")
            if self.observed_selection.fp != 0:
                raise ValueError("zero-observed-FP rule observed a false positive")
        else:
            if self.target_fpr is None:
                raise ValueError("FPR-budget rule requires target_fpr")
            counts = self.observed_selection
            n_negative = counts.fp + counts.tn
            count_derived_fpr = counts.fp / n_negative if n_negative else 0.0
            if count_derived_fpr > self.target_fpr:
                raise ValueError("observed selection FPR exceeds target_fpr")
        return self


class FrozenProtocol(BaseModel):
    """Validated encoding of the protocol frozen before Veritas was unsealed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    protocol_version: Literal[2]
    experiment_id: str
    model_fingerprint: str = Field(pattern=r"^pl1:[0-9a-f]{64}$")
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_bundle_fingerprint: str = Field(pattern=r"^pl1:[0-9a-f]{64}$")
    recipe_fingerprint: str = Field(pattern=r"^pl1:[0-9a-f]{64}$")
    execution_fingerprint: str = Field(pattern=r"^pl1:[0-9a-f]{64}$")
    calibrator_method: Literal["identity", "temperature", "platt", "isotonic"]
    calibrator_params: dict[str, Any]
    operating_points: dict[str, FrozenOperatingPoint]
    validation_n: int

    @model_validator(mode="after")
    def calibrator_contract_is_valid(self) -> FrozenProtocol:
        params = self.calibrator_params

        def require_exact_keys(expected: set[str]) -> None:
            if set(params) != expected:
                raise ValueError(f"calibrator parameters must contain exactly {sorted(expected)}")

        def finite_number(value: Any, *, name: str) -> float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"calibrator parameter {name} must be a finite number")
            try:
                converted = float(value)
            except (OverflowError, TypeError, ValueError) as exc:
                raise ValueError(f"calibrator parameter {name} must be a finite number") from exc
            if not math.isfinite(converted):
                raise ValueError(f"calibrator parameter {name} must be a finite number")
            return converted

        if self.calibrator_method == "identity":
            require_exact_keys(set())
            return self
        if self.calibrator_method == "temperature":
            require_exact_keys({"temperature"})
            if finite_number(params["temperature"], name="temperature") <= 0.0:
                raise ValueError("calibrator temperature must be positive")
            return self
        if self.calibrator_method == "platt":
            require_exact_keys({"a", "b"})
            finite_number(params["a"], name="a")
            finite_number(params["b"], name="b")
            return self

        require_exact_keys({"breakpoints", "iso_x", "iso_y"})
        breakpoints = params["breakpoints"]
        if isinstance(breakpoints, bool) or not isinstance(breakpoints, int) or breakpoints <= 0:
            raise ValueError("calibrator breakpoints must be a positive integer")
        iso_x_raw = params["iso_x"]
        iso_y_raw = params["iso_y"]
        if not isinstance(iso_x_raw, list) or not isinstance(iso_y_raw, list):
            raise ValueError("calibrator iso_x and iso_y must be arrays")
        if len(iso_x_raw) != breakpoints or len(iso_y_raw) != breakpoints:
            raise ValueError("calibrator breakpoint count must match iso_x and iso_y lengths")
        iso_x = [finite_number(value, name="iso_x") for value in iso_x_raw]
        iso_y = [finite_number(value, name="iso_y") for value in iso_y_raw]
        if any(not 0.0 <= value <= 1.0 for value in (*iso_x, *iso_y)):
            raise ValueError("calibrator isotonic values must be probabilities in [0, 1]")
        if any(left >= right for left, right in pairwise(iso_x)):
            raise ValueError("calibrator iso_x must be strictly increasing")
        if any(left > right for left, right in pairwise(iso_y)):
            raise ValueError("calibrator iso_y must be nondecreasing")
        return self

    @model_validator(mode="after")
    def operating_points_are_complete(self) -> FrozenProtocol:
        if self.model_fingerprint != self.recipe_fingerprint:
            raise ValueError("model_fingerprint must equal recipe_fingerprint")
        if set(self.operating_points) != OP_KEYS:
            raise ValueError(f"operating_points must contain exactly {sorted(OP_KEYS)}")
        class_counts: set[tuple[int, int]] = set()
        for key, point in self.operating_points.items():
            expected_rule, expected_budget = OP_SPECS[key]
            if point.selection_rule != expected_rule or point.target_fpr != expected_budget:
                raise ValueError(f"{key} does not match its required selection rule and budget")
            counts = point.observed_selection
            if counts.tp + counts.fp + counts.tn + counts.fn != self.validation_n:
                raise ValueError("operating-point counts do not sum to validation_n")
            class_counts.add((counts.tp + counts.fn, counts.fp + counts.tn))
        if len(class_counts) != 1:
            raise ValueError("operating points have inconsistent selection class counts")
        return self

    @property
    def artifact_identity(self) -> ArtifactIdentity:
        """The release identity this protocol and every derived report are bound to."""
        return ArtifactIdentity(
            checkpoint_sha256=self.checkpoint_sha256,
            artifact_bundle_fingerprint=self.artifact_bundle_fingerprint,
            recipe_fingerprint=self.recipe_fingerprint,
            execution_fingerprint=self.execution_fingerprint,
        )

    @property
    def thresholds(self) -> dict[str, float]:
        """Compatibility view for callers that only apply the frozen thresholds."""
        return {key: point.threshold for key, point in self.operating_points.items()}


def _op_key(budget: float) -> str:
    return "zero_fp" if budget == 0.0 else f"fpr_le_{budget}"


def derive_protocol(
    val_preds: list[BaselinePrediction],
    *,
    experiment_id: str,
    artifact_identity: ArtifactIdentity,
    isotonic_min_obs: int,
) -> FrozenProtocol:
    """Freeze thresholds + calibrator from VALIDATION only (never Veritas). Deterministic."""
    core = [p.core() for p in val_preds]
    scores = [p.score for p in val_preds]
    labels = [1 if p.gold == Label.CACHE_EQUIVALENT else 0 for p in val_preds]
    iso = calfit.fit_isotonic(scores, labels, min_obs=isotonic_min_obs)
    if iso is None:
        raise ValueError("isotonic calibrator could not be fit on validation")
    # Isotonic breakpoints live in the Calibrator's private fields, so serialize them in full — a
    # calibrator rebuilt from `params` alone would be a silent identity map (see _calibrator).
    cal_params = {
        "breakpoints": iso.params["breakpoints"],
        "iso_x": [round(x, 8) for x in iso._iso_x],
        "iso_y": [round(y, 8) for y in iso._iso_y],
    }
    operating_points: dict[str, FrozenOperatingPoint] = {}
    for budget in OP_BUDGETS:
        tau = (
            select.threshold_zero_fp(core)
            if budget == 0.0
            else select.threshold_at_fpr(core, budget)
        )
        c = metrics.confusion_at(core, tau)
        selection_metrics = SelectionMetrics(
            tp=c.tp,
            fp=c.fp,
            tn=c.tn,
            fn=c.fn,
            recall=round(c.tp / (c.tp + c.fn), 6) if c.tp + c.fn else 0.0,
            fpr=round(c.fp / (c.fp + c.tn), 6) if c.fp + c.tn else 0.0,
        )
        operating_points[_op_key(budget)] = FrozenOperatingPoint(
            selection_split="validation",
            selection_rule="zero_observed_fp" if budget == 0.0 else "fpr_budget",
            target_fpr=None if budget == 0.0 else budget,
            threshold=tau,
            observed_selection=selection_metrics,
        )
    return FrozenProtocol(
        protocol_version=2,
        experiment_id=experiment_id,
        model_fingerprint=artifact_identity.recipe_fingerprint,
        checkpoint_sha256=artifact_identity.checkpoint_sha256,
        artifact_bundle_fingerprint=artifact_identity.artifact_bundle_fingerprint,
        recipe_fingerprint=artifact_identity.recipe_fingerprint,
        execution_fingerprint=artifact_identity.execution_fingerprint,
        calibrator_method=iso.method,
        calibrator_params=cal_params,
        operating_points=operating_points,
        validation_n=len(val_preds),
    )


def _calibrator(protocol: FrozenProtocol) -> calfit.Calibrator:
    """Rebuild the frozen calibrator, restoring isotonic breakpoints from the protocol."""
    if protocol.calibrator_method == "isotonic":
        return calfit.Calibrator(
            method="isotonic",
            params={"breakpoints": protocol.calibrator_params["breakpoints"]},
            _iso_x=tuple(protocol.calibrator_params["iso_x"]),
            _iso_y=tuple(protocol.calibrator_params["iso_y"]),
        )
    return calfit.Calibrator(method=protocol.calibrator_method, params=protocol.calibrator_params)


def score_golden(
    checkpoint_dir: Path,
    cfg: TrainConfig,
    calibrator: calfit.Calibrator,
    *,
    golden_path: Path = GOLDEN_PATH,
    device: str = "cpu",
) -> list[GoldenScore]:
    """Author-run: score the 75 sealed pairs once with the frozen checkpoint (lazy torch)."""
    from polarity.baselines.materialize import MaterializedPair
    from polarity.data.loaders import load_golden
    from polarity.train import model as model_mod
    from polarity.train import runner
    from polarity.types import Category

    golden = load_golden(golden_path)
    pairs = [
        MaterializedPair(
            example_id=g.id,
            text_a=g.base,
            text_b=g.probe,
            gold=Label.CACHE_EQUIVALENT if g.equivalent else Label.NOT_CACHE_EQUIVALENT,
            category=Category.PARAPHRASE,  # unused here; slices use pairType below
            source="veritas-golden",
        )
        for g in golden
    ]
    bundle = model_mod.load_checkpoint(checkpoint_dir, cfg, device=device)
    raw = runner.predict_scores(bundle, pairs, cfg)
    calibrated = calibrator.apply(raw)
    return [
        GoldenScore(
            id=g.id,
            group=g.group,
            pair_type=g.pairType,
            gold_equivalent=g.equivalent,
            raw_score=round(float(s), 6),
            calibrated=round(float(c), 6),
        )
        for g, s, c in zip(golden, raw, calibrated, strict=True)
    ]


def validate_veritas_scoring_context(
    cfg: TrainConfig,
    protocol: FrozenProtocol,
    manifest: ArtifactReleaseContext,
) -> calfit.Calibrator:
    """Refuse any scoring request that differs from the frozen recipe/execution identity."""
    if protocol.artifact_identity != manifest.artifact_identity:
        raise ScoringPreflightError("frozen protocol identity does not match the artifact manifest")
    if protocol.thresholds != manifest.frozen_thresholds:
        raise ScoringPreflightError("threshold does not match the frozen protocol")
    if protocol.calibrator_method != manifest.calibrator_method:
        raise ScoringPreflightError("method does not match the frozen protocol")
    if cfg.experiment_id != protocol.experiment_id or cfg.experiment_id != manifest.experiment_id:
        raise ScoringPreflightError("experiment_id does not match the frozen protocol and manifest")
    if cfg.model.revision != manifest.execution.base_revision:
        raise ScoringPreflightError("base revision does not match the frozen execution identity")
    if cfg.model.tokenizer_revision != manifest.execution.tokenizer_revision:
        raise ScoringPreflightError(
            "tokenizer revision does not match the frozen execution identity"
        )
    actual_recipe = recipe_fingerprint(cfg)
    if actual_recipe != protocol.recipe_fingerprint or actual_recipe != (
        manifest.artifact_identity.recipe_fingerprint
    ):
        raise ScoringPreflightError(
            "recipe fingerprint does not match the frozen protocol and manifest"
        )
    return _calibrator(protocol)


def _as_predictions(scores: list[GoldenScore]) -> list[Prediction]:
    from polarity.types import Category

    return [
        Prediction(
            pair_id=s.id,
            gold=Label.CACHE_EQUIVALENT if s.gold_equivalent else Label.NOT_CACHE_EQUIVALENT,
            score=s.raw_score,
            category=Category.PARAPHRASE,
            source="veritas-golden",
        )
        for s in scores
    ]


def _op_metrics(preds: list[Prediction], tau: float) -> dict[str, Any]:
    c = metrics.confusion_at(preds, tau)
    recall = c.tp / (c.tp + c.fn) if (c.tp + c.fn) else 0.0
    fpr = c.fp / (c.fp + c.tn) if (c.fp + c.tn) else 0.0
    precision = c.tp / (c.tp + c.fp) if (c.tp + c.fp) else 1.0
    return {
        "tau": tau,
        "tp": c.tp,
        "fp": c.fp,
        "tn": c.tn,
        "fn": c.fn,
        "recall": round(recall, 6),
        "fpr": round(fpr, 6),
        "precision": round(precision, 6),
        "false_positives": c.fp,
    }


def _slice_metrics(scores: list[GoldenScore], tau: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pt in sorted({s.pair_type for s in scores}):
        items = [s for s in scores if s.pair_type == pt]
        pos = [s for s in items if s.gold_equivalent]
        neg = [s for s in items if not s.gold_equivalent]
        hit = sum(1 for s in pos if s.raw_score >= tau)
        fp = sum(1 for s in neg if s.raw_score >= tau)
        out[pt] = {
            "n": len(items),
            "n_positive": len(pos),
            "n_negative": len(neg),
            "recall": round(hit / len(pos), 6) if pos else None,
            "false_positives": fp,
            "fp_rate": round(fp / len(neg), 6) if neg else None,
        }
    return out


def transfer_report(scores: list[GoldenScore], protocol: FrozenProtocol) -> dict[str, Any]:
    """Keyless: recompute the full transfer verdict from the committed scores + frozen protocol."""
    preds = _as_predictions(scores)
    n_pos = sum(1 for s in scores if s.gold_equivalent)
    n_neg = len(scores) - n_pos
    cal_scores = [s.calibrated for s in scores]
    labels = [1 if s.gold_equivalent else 0 for s in scores]
    operating = {}
    for key, point in protocol.operating_points.items():
        tau = point.threshold
        operating[key] = {
            "selection": point.model_dump(mode="json"),
            "realized": {
                "split": "veritas",
                "overall": _op_metrics(preds, tau),
                "by_pair_type": _slice_metrics(scores, tau),
            },
        }
    threshold_free_tau = select.threshold_at_fpr(preds, 0.01)
    return {
        "protocol_version": protocol.protocol_version,
        "experiment_id": protocol.experiment_id,
        "checkpoint_sha256": protocol.checkpoint_sha256,
        "model_fingerprint": protocol.model_fingerprint,
        "artifact_bundle_fingerprint": protocol.artifact_bundle_fingerprint,
        "recipe_fingerprint": protocol.recipe_fingerprint,
        "execution_fingerprint": protocol.execution_fingerprint,
        "n_pairs": len(scores),
        "n_equivalent": n_pos,
        "n_not_equivalent": n_neg,
        "fp_definition": metrics.FP_DEFINITION,
        "frozen_thresholds": protocol.thresholds,
        "operating_points": operating,
        "threshold_free": {
            "recall_at_fpr_le_0.01": {
                "split": "veritas",
                "target_fpr": 0.01,
                "metric": "threshold-free recall at actual dataset FPR <= 0.01",
                "overall": _op_metrics(preds, threshold_free_tau),
            }
        },
        "calibration": {
            "method": protocol.calibrator_method,
            "ece_raw": round(expected_calibration_error([s.raw_score for s in scores], labels), 6),
            "ece_calibrated": round(expected_calibration_error(cal_scores, labels), 6),
        },
    }


def reproduction_classify(
    new_preds: list[BaselinePrediction],
    committed: list[BaselinePrediction],
    *,
    protocol: FrozenProtocol,
    score_tol: float = 0.02,
    recall_tol: float = 0.03,
) -> dict[str, Any]:
    """Compare candidate and committed M3.3 scores after one-to-one ID validation.

    The strongest verdict means only that every aligned per-row score differs by less than 1e-6.
    It does not establish checkpoint identity or fully reproducible retraining.
    """

    if not new_preds:
        raise ReproductionError("candidate predictions are empty")
    if not committed:
        raise ReproductionError("committed predictions are empty")

    def _duplicates(preds: list[BaselinePrediction]) -> list[str]:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for pred in preds:
            if pred.pair_id in seen:
                duplicates.add(pred.pair_id)
            seen.add(pred.pair_id)
        return sorted(duplicates)

    for name, preds in (("candidate", new_preds), ("committed", committed)):
        duplicates = _duplicates(preds)
        if duplicates:
            return {
                "verdict": "MATERIAL_REPRODUCTION_DRIFT",
                "reason": f"duplicate pair_id in {name} predictions: {','.join(duplicates)}",
            }

    candidate_by_id = {p.pair_id: p for p in new_preds}
    committed_by_id = {p.pair_id: p for p in committed}
    candidate_ids = set(candidate_by_id)
    committed_ids = set(committed_by_id)
    if candidate_ids != committed_ids:
        missing = ",".join(sorted(committed_ids - candidate_ids)) or "none"
        unexpected = ",".join(sorted(candidate_ids - committed_ids)) or "none"
        return {
            "verdict": "MATERIAL_REPRODUCTION_DRIFT",
            "reason": (
                "pair-id set differs; "
                f"missing_from_candidate={missing}; unexpected_in_candidate={unexpected}"
            ),
        }

    aligned = [(candidate_by_id[p.pair_id], p) for p in committed]
    diffs = [abs(n.score - c.score) for n, c in aligned]
    max_diff = max(diffs)
    mean_diff = sum(diffs) / len(diffs)
    # Compare only score drift: labels and other metric metadata come from the committed rows.
    new_core = [
        Prediction(
            pair_id=committed_pred.pair_id,
            gold=committed_pred.gold,
            score=new_pred.score,
            category=committed_pred.category,
            source=committed_pred.source,
        )
        for new_pred, committed_pred in aligned
    ]
    com_core = [p.core() for p in committed]

    def _recall_at(core: list[Prediction], tau: float) -> float:
        c = metrics.confusion_at(core, tau)
        return c.tp / (c.tp + c.fn) if (c.tp + c.fn) else 0.0

    operating_point = protocol.operating_points["fpr_le_0.01"]
    tau = operating_point.threshold
    recall_delta = abs(_recall_at(new_core, tau) - _recall_at(com_core, tau))
    score_agreement_tolerance = 1e-6
    score_agreement = max_diff < score_agreement_tolerance
    if score_agreement:
        verdict = "SCORE_AGREEMENT_WITHIN_TOLERANCE"
    elif max_diff <= score_tol and recall_delta <= recall_tol:
        verdict = "METRIC_REPRODUCTION_WITH_NUMERICAL_VARIATION"
    else:
        verdict = "MATERIAL_REPRODUCTION_DRIFT"
    return {
        "verdict": verdict,
        "n": len(aligned),
        "max_abs_score_diff": max_diff,
        "mean_abs_score_diff": mean_diff,
        "protocol_version": protocol.protocol_version,
        "experiment_id": protocol.experiment_id,
        "operating_point": "fpr_le_0.01",
        "selection_split": operating_point.selection_split,
        "target_fpr": operating_point.target_fpr,
        "frozen_validation_selected_threshold": tau,
        "recall_delta_at_frozen_validation_selected_threshold": recall_delta,
        "score_agreement_tolerance": score_agreement_tolerance,
        "score_agreement_rule": "aligned per-row absolute score difference < 1e-6",
        "evidence_scope": (
            "score agreement only; does not establish checkpoint identity or retraining "
            "reproducibility"
        ),
        "score_tol": score_tol,
        "recall_tol": recall_tol,
    }


def write_scores(scores: list[GoldenScore], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [s.model_dump_json() for s in sorted(scores, key=lambda x: x.id)]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_scores(path: Path) -> list[GoldenScore]:
    return [
        GoldenScore.model_validate_json(ln)
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]


def load_preds(path: Path, *, kind: str = "predictions") -> list[BaselinePrediction]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ReproductionError(f"cannot read {kind} at {path}: {exc}") from exc
    predictions: list[BaselinePrediction] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            predictions.append(BaselinePrediction.model_validate_json(line))
        except ValidationError as exc:
            raise ReproductionError(f"invalid {kind} at {path}, line {line_number}: {exc}") from exc
    return predictions


def load_protocol(path: Path, *, expected_sha256: str | None = None) -> FrozenProtocol:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ReproductionError(f"cannot read frozen protocol at {path}: {exc}") from exc
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ReproductionError(
            "protocol sha256 does not match the artifact manifest: "
            f"{actual_sha256} != {expected_sha256}"
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise ReproductionError(f"cannot read frozen protocol at {path}: {exc}") from exc
    try:
        return FrozenProtocol.model_validate_json(text)
    except ValidationError as exc:
        raise ReproductionError(f"invalid frozen protocol at {path}: {exc}") from exc


def dump(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, indent=2) + "\n"


def checkpoint_sha256(checkpoint_dir: Path) -> str:
    """sha256 of the checkpoint weights (model.safetensors) — the exact-artifact identity (§2)."""
    import hashlib

    weights = checkpoint_dir / "model.safetensors"
    h = hashlib.sha256()
    with weights.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_report_files(scores: list[GoldenScore], protocol: FrozenProtocol) -> dict[str, str]:
    """The committed, drift-gated M6 artifacts, keyed by filename."""
    rep = transfer_report(scores, protocol)
    return {f"transfer-{protocol.experiment_id}.json": dump(rep)}


def check_report(
    scores: list[GoldenScore], protocol: FrozenProtocol, out_root: Path = Path(".")
) -> list[str]:
    drift: list[str] = []
    for fname, text in build_report_files(scores, protocol).items():
        path = out_root / REPORTS_ROOT / fname
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            drift.append(fname)
    return drift


__all__ = [
    "FrozenOperatingPoint",
    "FrozenProtocol",
    "GoldenScore",
    "ReproductionError",
    "ScoringPreflightError",
    "SelectionMetrics",
    "build_report_files",
    "check_report",
    "checkpoint_sha256",
    "derive_protocol",
    "load_protocol",
    "load_scores",
    "reproduction_classify",
    "score_golden",
    "transfer_report",
    "validate_veritas_scoring_context",
    "write_scores",
]
