"""Operational metrics for cache equivalence.

Positive class = ``CACHE_EQUIVALENT``. A **false positive** is a ``NOT_CACHE_EQUIVALENT``
pair predicted ``CACHE_EQUIVALENT`` (score >= tau) — the failure that serves a wrong cached
answer, and the one Polarity must keep rare. The primary curve is therefore recall vs.
false-positive rate, not accuracy or AUC.

Determinism (drift gate): every serialized number is rounded to ``ROUND_DP`` decimals, keys
are sorted, and the payload carries no timestamps, paths, or RNG-dependent ordering, so
``polarity evaluate --check`` reproduces the committed report byte-for-byte across machines
and across the 3.12/3.13 CI matrix.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from polarity.types import Category, Label, Prediction

ROUND_DP = 6
REPORT_SCHEMA = "polarity-metrics-1"
POSITIVE_CLASS = Label.CACHE_EQUIVALENT
FP_DEFINITION = (
    "false_positive = gold NOT_CACHE_EQUIVALENT predicted CACHE_EQUIVALENT (score >= tau)"
)
_DEFAULT_GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def _r(x: float) -> float:
    return round(x, ROUND_DP)


def _is_positive(gold: Label) -> bool:
    return gold == Label.CACHE_EQUIVALENT


class ConfusionCounts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tp: int
    fp: int
    tn: int
    fn: int


class ThresholdRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tau: float
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    fpr: float
    f1: float


class OperatingPoint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    recall: float
    fp: int
    n_negatives: int
    max_negative_score: float
    note: str


class CategoryError(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    category: Category
    n: int
    errors: int
    error_rate: float


class RecallCI(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tau: float
    recall: float
    ci_low: float
    ci_high: float
    n_boot: int
    seed: int


class MetricsReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str
    positive_class: str
    fp_definition: str
    n: int
    n_positive: int
    n_negative: int
    best_f1: ThresholdRow
    recall_at_zero_fp: OperatingPoint
    recall_ci: RecallCI
    threshold_table: list[ThresholdRow]
    per_category: list[CategoryError]
    fpr_resolution_note: str


def confusion_at(preds: list[Prediction], tau: float) -> ConfusionCounts:
    tp = fp = tn = fn = 0
    for p in preds:
        pos = _is_positive(p.gold)
        pred_pos = p.score >= tau
        if pos and pred_pos:
            tp += 1
        elif pos and not pred_pos:
            fn += 1
        elif not pos and pred_pos:
            fp += 1
        else:
            tn += 1
    return ConfusionCounts(tp=tp, fp=fp, tn=tn, fn=fn)


def _row(preds: list[Prediction], tau: float) -> ThresholdRow:
    c = confusion_at(preds, tau)
    precision = c.tp / (c.tp + c.fp) if (c.tp + c.fp) else 1.0
    recall = c.tp / (c.tp + c.fn) if (c.tp + c.fn) else 0.0
    fpr = c.fp / (c.fp + c.tn) if (c.fp + c.tn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return ThresholdRow(
        tau=_r(tau),
        tp=c.tp,
        fp=c.fp,
        tn=c.tn,
        fn=c.fn,
        precision=_r(precision),
        recall=_r(recall),
        fpr=_r(fpr),
        f1=_r(f1),
    )


def sweep(preds: list[Prediction], taus: list[float]) -> list[ThresholdRow]:
    return [_row(preds, t) for t in taus]


def best_f1_threshold(preds: list[Prediction]) -> ThresholdRow:
    """Best single-threshold F1 over the observed score grid (ties broken by lower tau)."""
    candidates = sorted({p.score for p in preds})
    if not candidates:
        raise ValueError("no predictions")
    rows = [_row(preds, t) for t in candidates]
    return max(rows, key=lambda r: (r.f1, -r.tau))


def recall_at_zero_fp(preds: list[Prediction]) -> OperatingPoint:
    """Recall achievable while admitting zero observed false positives.

    A threshold gives 0 FP only if it exceeds every negative's score. Recall is then the
    fraction of positives scoring strictly above the top negative. With few negatives this
    is a coarse, finite-sample statement — see ``fpr_resolution_note``.
    """
    pos = [p.score for p in preds if _is_positive(p.gold)]
    neg = [p.score for p in preds if not _is_positive(p.gold)]
    n_pos = len(pos)
    if not neg:
        recall = 1.0 if n_pos else 0.0
        return OperatingPoint(
            name="recall_at_0_observed_fp",
            recall=_r(recall),
            fp=0,
            n_negatives=0,
            max_negative_score=0.0,
            note="no negatives in set",
        )
    max_neg = max(neg)
    admitted = sum(1 for s in pos if s > max_neg)
    recall = admitted / n_pos if n_pos else 0.0
    note = (
        f"threshold must exceed top negative ({_r(max_neg)}); {admitted}/{n_pos} positives clear it"
    )
    return OperatingPoint(
        name="recall_at_0_observed_fp",
        recall=_r(recall),
        fp=0,
        n_negatives=len(neg),
        max_negative_score=_r(max_neg),
        note=note,
    )


def per_category(preds: list[Prediction], tau: float) -> list[CategoryError]:
    buckets: dict[Category, list[Prediction]] = {}
    for p in preds:
        buckets.setdefault(p.category, []).append(p)
    rows: list[CategoryError] = []
    for cat in sorted(buckets, key=lambda c: c.value):
        items = buckets[cat]
        errors = sum(1 for p in items if (p.score >= tau) != _is_positive(p.gold))
        rows.append(
            CategoryError(
                category=cat,
                n=len(items),
                errors=errors,
                error_rate=_r(errors / len(items)) if items else 0.0,
            )
        )
    return rows


def bootstrap_recall_ci(
    preds: list[Prediction], tau: float, *, n_boot: int = 1000, seed: int = 42
) -> RecallCI:
    """Percentile bootstrap CI for recall at a fixed tau. Seeded, so it is reproducible."""
    positives = [p for p in preds if _is_positive(p.gold)]
    n = len(positives)
    base = sum(1 for p in positives if p.score >= tau) / n if n else 0.0
    if n == 0:
        return RecallCI(tau=_r(tau), recall=0.0, ci_low=0.0, ci_high=0.0, n_boot=n_boot, seed=seed)
    rng = random.Random(seed)
    hits = [1 if p.score >= tau else 0 for p in positives]
    samples: list[float] = []
    for _ in range(n_boot):
        draw = sum(hits[rng.randrange(n)] for _ in range(n))
        samples.append(draw / n)
    samples.sort()
    lo = samples[int(0.025 * (n_boot - 1))]
    hi = samples[int(0.975 * (n_boot - 1))]
    return RecallCI(
        tau=_r(tau), recall=_r(base), ci_low=_r(lo), ci_high=_r(hi), n_boot=n_boot, seed=seed
    )


def build_report(preds: list[Prediction], *, ci_tau: float = 0.5) -> MetricsReport:
    n_pos = sum(1 for p in preds if _is_positive(p.gold))
    n_neg = len(preds) - n_pos
    resolution = (
        f"{n_neg} negatives => FPR granularity 1/{n_neg} "
        f"= {_r(1 / n_neg) if n_neg else 0.0}; sub-granularity FPR points are not resolvable. "
        "Report '0 observed FP on N negatives', not a population 0% FPR."
    )
    return MetricsReport(
        schema_version=REPORT_SCHEMA,
        positive_class=POSITIVE_CLASS.value,
        fp_definition=FP_DEFINITION,
        n=len(preds),
        n_positive=n_pos,
        n_negative=n_neg,
        best_f1=best_f1_threshold(preds),
        recall_at_zero_fp=recall_at_zero_fp(preds),
        recall_ci=bootstrap_recall_ci(preds, ci_tau),
        threshold_table=sweep(preds, list(_DEFAULT_GRID)),
        per_category=per_category(preds, ci_tau),
        fpr_resolution_note=resolution,
    )


def serialize_report(report: MetricsReport) -> str:
    """Deterministic, human-diffable JSON: sorted keys, 2-space indent, trailing newline."""
    return json.dumps(report.model_dump(mode="json"), sort_keys=True, indent=2) + "\n"


def load_predictions(path: Path) -> list[Prediction]:
    preds: list[Prediction] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            preds.append(Prediction.model_validate_json(line))
    return preds


def check_report(preds: list[Prediction], report_path: Path) -> bool:
    """True iff the committed report byte-for-byte matches a fresh recomputation."""
    return serialize_report(build_report(preds)) == report_path.read_text(encoding="utf-8")


__all__ = [
    "FP_DEFINITION",
    "MetricsReport",
    "build_report",
    "check_report",
    "confusion_at",
    "load_predictions",
    "per_category",
    "recall_at_zero_fp",
    "serialize_report",
    "sweep",
]
