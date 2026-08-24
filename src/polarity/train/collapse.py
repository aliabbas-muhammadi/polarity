"""Collapse diagnostics for the M3.2 fp32 run (brief §4). Pure, torch-free, unit-tested.

M3/M3.1 (bf16) collapsed to a (near-)constant output: the model learned the class prior and
nothing input-dependent, so no threshold separated anything and recall was 0 at every FP budget
(docs/M4-REPORT.md). M3.2 must first clear a *mechanics* gate — produce distinct, input-dependent
scores — before its low-FP numbers mean anything. These functions turn a set of scored, labelled
predictions into a collapse verdict against thresholds frozen *before* the run (``CollapseCfg``).
The runner evaluates them on the VALIDATION scores at each checkpoint and hard-stops on sustained
collapse; the keyless report recomputes the same verdict from the committed predictions.

Score = P(CACHE_EQUIVALENT); a healthy model spreads scores and separates the classes (the NLI
baseline: positive median ~0.99, negative median ~0.0003). A collapsed model does neither.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from polarity.train.config import CollapseCfg
from polarity.types import Label, Prediction

# Distinct-score count is measured at the 6dp precision the predictions are written with, so the
# runtime verdict and the keyless report agree byte-for-byte on what "constant" means.
SCORE_DP = 6


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))


@dataclass(frozen=True)
class ScoreStats:
    """Distribution summary of one split's scores. ``mean_pos``/``mean_neg`` are None if a class
    is absent (then ``class_separation`` is None too)."""

    n: int
    std: float
    n_unique: int
    mean_pos: float | None
    mean_neg: float | None
    class_separation: float | None  # abs(mean_pos - mean_neg)
    smin: float
    smax: float
    has_nonfinite: bool

    def as_dict(self) -> dict[str, float | int | bool | None]:
        def r(x: float | None) -> float | None:
            return round(x, SCORE_DP) if x is not None else None

        return {
            "n": self.n,
            "std": r(self.std),
            "n_unique": self.n_unique,
            "mean_pos": r(self.mean_pos),
            "mean_neg": r(self.mean_neg),
            "class_separation": r(self.class_separation),
            "min": r(self.smin),
            "max": r(self.smax),
            "has_nonfinite": self.has_nonfinite,
        }


def score_stats(scores: list[float], is_positive: list[bool]) -> ScoreStats:
    """Core: distribution stats from parallel score / class lists. No torch, no dependencies."""
    if len(scores) != len(is_positive):
        raise ValueError("scores and is_positive must be the same length")
    has_nonfinite = any(not math.isfinite(s) for s in scores)
    finite = [s for s in scores if math.isfinite(s)]
    if not finite:
        return ScoreStats(len(scores), 0.0, 0, None, None, None, 0.0, 0.0, has_nonfinite)
    pos = [s for s, p in zip(scores, is_positive, strict=True) if p and math.isfinite(s)]
    neg = [s for s, p in zip(scores, is_positive, strict=True) if not p and math.isfinite(s)]
    mean_pos = _mean(pos) if pos else None
    mean_neg = _mean(neg) if neg else None
    sep = abs(mean_pos - mean_neg) if (mean_pos is not None and mean_neg is not None) else None
    return ScoreStats(
        n=len(scores),
        std=_std(finite),
        n_unique=len({round(s, SCORE_DP) for s in finite}),
        mean_pos=mean_pos,
        mean_neg=mean_neg,
        class_separation=sep,
        smin=min(finite),
        smax=max(finite),
        has_nonfinite=has_nonfinite,
    )


def stats_from_predictions(preds: list[Prediction]) -> ScoreStats:
    return score_stats([p.score for p in preds], [p.gold == Label.CACHE_EQUIVALENT for p in preds])


@dataclass(frozen=True)
class CollapseVerdict:
    collapsed: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {"collapsed": self.collapsed, "reasons": list(self.reasons)}


def assess(stats: ScoreStats, cfg: CollapseCfg) -> CollapseVerdict:
    """Verdict from a single split's stats against the pre-registered thresholds.

    Collapsed if ANY holds: non-finite scores; fewer than ``min_unique_scores`` distinct values;
    score std below ``min_score_std``; or (when both classes are present) class separation below
    ``min_class_separation``. Class separation is the load-bearing signal — a constant-output model
    gives positives and negatives the same mean, so no threshold discriminates.
    """
    reasons: list[str] = []
    if stats.has_nonfinite:
        reasons.append("non-finite scores")
    if stats.n_unique < cfg.min_unique_scores:
        reasons.append(f"n_unique {stats.n_unique} < {cfg.min_unique_scores}")
    if stats.std < cfg.min_score_std:
        reasons.append(f"score std {stats.std:.2e} < {cfg.min_score_std:.2e}")
    if stats.class_separation is not None and stats.class_separation < cfg.min_class_separation:
        reasons.append(
            f"class separation {stats.class_separation:.2e} < {cfg.min_class_separation:.2e}"
        )
    return CollapseVerdict(collapsed=bool(reasons), reasons=tuple(reasons))


def is_finite(x: float) -> bool:
    """Guard for loss / grad-norm telemetry — a non-finite value is an instant training failure."""
    return math.isfinite(x)


def demo() -> None:
    cfg = CollapseCfg()
    # Constant output (M3.1's failure): one distinct value, zero separation -> collapsed.
    const = score_stats([0.488] * 100, [i % 2 == 0 for i in range(100)])
    assert assess(const, cfg).collapsed, const
    assert const.n_unique == 1 and const.class_separation == 0.0
    # Healthy separable model: positives high, negatives low -> not collapsed.
    scores = [0.98 if i % 2 == 0 else 0.02 for i in range(100)]
    healthy = score_stats(scores, [i % 2 == 0 for i in range(100)])
    assert not assess(healthy, cfg).collapsed, healthy
    # Non-finite is an instant collapse.
    assert assess(score_stats([float("nan"), 0.5], [True, False]), cfg).collapsed
    assert not is_finite(float("inf"))
    print("train.collapse demo OK")


__all__ = [
    "SCORE_DP",
    "CollapseVerdict",
    "ScoreStats",
    "assess",
    "is_finite",
    "score_stats",
    "stats_from_predictions",
]


if __name__ == "__main__":
    demo()
