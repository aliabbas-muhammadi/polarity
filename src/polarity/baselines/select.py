"""Operating-point selection on VALIDATION only; frozen thresholds are applied to internal_test.

Reuses the metric primitives in ``eval/metrics.py``. Keyless and deterministic.
"""

from __future__ import annotations

import math

from polarity.eval import metrics
from polarity.types import Label, Prediction


def threshold_zero_fp(validation: list[Prediction]) -> float:
    """Lowest threshold with 0 FP on validation = just above the top negative score."""
    neg = [p.score for p in validation if p.gold == Label.NOT_CACHE_EQUIVALENT]
    return math.nextafter(max(neg), 2.0) if neg else 0.0


def threshold_at_fpr(validation: list[Prediction], max_fpr: float) -> float:
    """Lowest threshold whose validation FPR <= max_fpr (maximises recall under the budget)."""
    for tau in sorted({p.score for p in validation}):
        c = metrics.confusion_at(validation, tau)
        fpr = c.fp / (c.fp + c.tn) if (c.fp + c.tn) else 0.0
        if fpr <= max_fpr:
            return tau
    return math.nextafter(1.0, 2.0)


__all__ = ["threshold_at_fpr", "threshold_zero_fp"]
