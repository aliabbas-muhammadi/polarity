"""FP-recall (ROC-in-the-low-FP-corner) curve + a correct partial AUC. Pure, deterministic.

M3/M3.1 used a buggy pAUC that linearly interpolated recall across the [0, fmax] band whenever the
first false-positive operating point landed *past* fmax — crediting recall the model never achieves
at low FPR, so a degenerate "reject-everything-then-accept-a-burst" curve scored ~0.5 (see
docs/M3.1-REPORT.md). This module replaces it with a **step** integral: the recall achievable at a
false-positive budget b is the recall of the best operating point with FPR <= b, and nothing is
credited beyond the last such point. A degenerate all-or-nothing curve therefore scores ~0, as it
should. Positive = CACHE_EQUIVALENT; a false positive is a NOT pair predicted positive.
"""

from __future__ import annotations

from polarity.eval.metrics import confusion_at
from polarity.types import Prediction


def roc_points(preds: list[Prediction]) -> list[tuple[float, float]]:
    """(_fpr, recall) operating points, one per distinct score, collapsed to max recall per FPR.

    Includes the ``(0.0, recall@0FP)`` anchor. Sorted by FPR ascending; recall is non-decreasing.
    """
    scores = sorted({p.score for p in preds}, reverse=True)
    best: dict[float, float] = {0.0: 0.0}
    for tau in scores:
        c = confusion_at(preds, tau)
        fpr = c.fp / (c.fp + c.tn) if (c.fp + c.tn) else 0.0
        rec = c.tp / (c.tp + c.fn) if (c.tp + c.fn) else 0.0
        best[fpr] = max(best.get(fpr, 0.0), rec)
    return sorted(best.items())


def partial_auc(preds: list[Prediction], fmax: float) -> float:
    """Step-integrated partial AUC of the FP-recall curve over FPR in [0, fmax], normalized to 1.

    recall_achievable(b) = max recall over operating points with FPR <= b; no recall is credited
    beyond the last operating point at or below fmax (an all-or-nothing tail scores ~0, not ~0.5).
    """
    if fmax <= 0:
        return 0.0
    pts = roc_points(preds)
    area = 0.0
    prev_f = 0.0
    prev_r = 0.0  # recall achievable at FPR 0 (updated from the (0.0, .) anchor below)
    for f, r in pts:
        if f <= 0.0:
            prev_r = max(prev_r, r)
            continue
        seg_end = min(f, fmax)
        if seg_end > prev_f:
            area += prev_r * (seg_end - prev_f)  # achievable recall in [prev_f, f) is prev_r
            prev_f = seg_end
        prev_r = max(prev_r, r)
        if prev_f >= fmax:
            break
    if prev_f < fmax:
        area += prev_r * (fmax - prev_f)
    return round(area / fmax, 6)


def recall_at_fpr_budgets(preds: list[Prediction], budgets: tuple[float, ...]) -> dict[str, float]:
    """Max achievable recall at each FPR budget (best operating point with FPR <= budget)."""
    pts = roc_points(preds)
    out: dict[str, float] = {}
    for b in budgets:
        out[str(b)] = round(max((r for f, r in pts if f <= b + 1e-12), default=0.0), 6)
    return out


def demo() -> None:
    """Self-check: an all-or-nothing tail scores ~0; a graded low-FP curve scores > 0."""
    from polarity.types import Category, Label

    def _p(pid: str, gold: str, score: float) -> Prediction:
        return Prediction(
            pair_id=pid,
            gold=Label(gold),
            score=score,
            category=Category.PARAPHRASE,
            source="x",
        )

    # Degenerate: every negative scores 1.0 (a burst past any low FPR); positives lower.
    deg = [_p(f"n{i}", "NOT_CACHE_EQUIVALENT", 1.0) for i in range(100)]
    deg += [_p(f"p{i}", "CACHE_EQUIVALENT", 0.9) for i in range(100)]
    assert partial_auc(deg, 0.02) == 0.0, partial_auc(deg, 0.02)

    # Separable: positives (1.0) outrank every negative (<0.5), so recall@0FP is high.
    grad = [_p(f"n{i}", "NOT_CACHE_EQUIVALENT", i / 200.0) for i in range(100)]
    grad += [_p(f"p{i}", "CACHE_EQUIVALENT", 1.0) for i in range(100)]
    assert partial_auc(grad, 0.02) > 0.5, partial_auc(grad, 0.02)
    # Monotone in fmax: a wider band never decreases the (un-normalized) achievable area intuition —
    # here recall@0FP is 1.0 so pAUC is 1.0 at any band.
    assert recall_at_fpr_budgets(grad, (0.01,))["0.01"] == 1.0
    print("eval.curves demo OK")


__all__ = ["partial_auc", "recall_at_fpr_budgets", "roc_points"]


if __name__ == "__main__":
    demo()
