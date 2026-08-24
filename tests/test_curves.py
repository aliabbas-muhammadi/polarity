from polarity.eval.curves import partial_auc, recall_at_fpr_budgets, roc_points
from polarity.types import Category, Label, Prediction


def _p(pid: str, gold: str, score: float) -> Prediction:
    return Prediction(
        pair_id=pid, gold=Label(gold), score=score, category=Category.PARAPHRASE, source="x"
    )


def _degenerate() -> list[Prediction]:
    # all-or-nothing tail: every negative scores 1.0, positives lower -> no low-FP recall
    neg = [_p(f"n{i}", "NOT_CACHE_EQUIVALENT", 1.0) for i in range(100)]
    pos = [_p(f"p{i}", "CACHE_EQUIVALENT", 0.9) for i in range(100)]
    return neg + pos


def _separable() -> list[Prediction]:
    neg = [_p(f"n{i}", "NOT_CACHE_EQUIVALENT", i / 200.0) for i in range(100)]  # < 0.5
    pos = [_p(f"p{i}", "CACHE_EQUIVALENT", 1.0) for i in range(100)]
    return neg + pos


def test_degenerate_tail_scores_zero_not_half():
    # the M3.1 bug: the old _pauc credited this ~0.5; the fix must return ~0.
    assert partial_auc(_degenerate(), 0.02) == 0.0


def test_separable_scores_high():
    assert partial_auc(_separable(), 0.02) > 0.5


def test_collapsed_constant_scores_zero():
    # constant output (the observed M3/M3.1 failure) -> no discrimination -> pAUC 0
    rows = [_p(f"n{i}", "NOT_CACHE_EQUIVALENT", 0.47) for i in range(50)]
    rows += [_p(f"p{i}", "CACHE_EQUIVALENT", 0.47) for i in range(50)]
    assert partial_auc(rows, 0.02) == 0.0
    assert recall_at_fpr_budgets(rows, (0.01, 0.05))["0.05"] == 0.0


def test_pauc_bounded_and_monotone_in_fmax():
    preds = _separable()
    a1 = partial_auc(preds, 0.01)
    a2 = partial_auc(preds, 0.05)
    assert 0.0 <= a1 <= 1.0 and 0.0 <= a2 <= 1.0
    # perfectly separable (recall@0FP = 1.0) -> pAUC 1.0 at any band
    assert a1 == 1.0 and a2 == 1.0


def test_recall_budgets_monotone_nondecreasing():
    preds = _separable()
    r = recall_at_fpr_budgets(preds, (0.005, 0.01, 0.02, 0.05))
    vals = [r["0.005"], r["0.01"], r["0.02"], r["0.05"]]
    assert vals == sorted(vals)  # wider budget never lowers achievable recall


def test_roc_points_start_at_origin_and_sorted():
    pts = roc_points(_separable())
    assert pts[0][0] == 0.0
    assert pts == sorted(pts)
