"""Calibration-metric foundations.

Polarity's whole deliverable is a *thresholded* accept/reject at a chosen false-positive
budget, and the literature (docs/LITERATURE.md) shows a cross-encoder can have excellent
ranking AUC yet be badly miscalibrated at the deployment threshold. So calibration is not
optional. M4 fits the calibrator (temperature / isotonic / Platt) on validation data; M0
ships the two metrics used to judge it — the Brier score and expected calibration error —
implemented and tested here. Labels are 1 for the positive class (CACHE_EQUIVALENT).
"""

from __future__ import annotations

from collections.abc import Sequence


def brier_score(probs: Sequence[float], labels: Sequence[int]) -> float:
    """Mean squared error between predicted probability and outcome."""
    if len(probs) != len(labels):
        raise ValueError("probs and labels must be the same length")
    if not probs:
        raise ValueError("empty input")
    return sum((p - y) ** 2 for p, y in zip(probs, labels, strict=True)) / len(probs)


def expected_calibration_error(
    probs: Sequence[float], labels: Sequence[int], *, n_bins: int = 10
) -> float:
    """Binned |confidence - accuracy|, weighted by bin population (ECE).

    Bins are equal-width over [0, 1]; a probability of exactly 1.0 falls in the top bin.
    """
    if len(probs) != len(labels):
        raise ValueError("probs and labels must be the same length")
    if not probs:
        raise ValueError("empty input")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    total = len(probs)
    ece = 0.0
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        # Include the right edge only in the final bin so p == 1.0 is counted once.
        members = [
            (p, y)
            for p, y in zip(probs, labels, strict=True)
            if (lo <= p < hi) or (b == n_bins - 1 and p == hi)
        ]
        if not members:
            continue
        conf = sum(p for p, _ in members) / len(members)
        acc = sum(y for _, y in members) / len(members)
        ece += (len(members) / total) * abs(conf - acc)
    return ece


__all__ = ["brier_score", "expected_calibration_error"]
