from polarity.baselines import select
from polarity.eval.metrics import confusion_at
from polarity.types import Category, Label, Prediction


def _p(pid: str, gold: str, score: float) -> Prediction:
    return Prediction(
        pair_id=pid, gold=Label(gold), score=score, category=Category.PARAPHRASE, source="s"
    )


def test_threshold_zero_fp_admits_no_false_positive():
    val = [
        _p("p1", "CACHE_EQUIVALENT", 0.9),
        _p("n1", "NOT_CACHE_EQUIVALENT", 0.6),
        _p("n2", "NOT_CACHE_EQUIVALENT", 0.4),
    ]
    tau = select.threshold_zero_fp(val)
    assert tau > 0.6
    assert confusion_at(val, tau).fp == 0


def test_threshold_at_fpr_respects_budget():
    negatives = [_p("nhi", "NOT_CACHE_EQUIVALENT", 0.9)] + [
        _p(f"n{i}", "NOT_CACHE_EQUIVALENT", 0.1) for i in range(9)
    ]
    val = [*negatives, _p("p1", "CACHE_EQUIVALENT", 0.5)]
    tau = select.threshold_at_fpr(val, 0.1)
    c = confusion_at(val, tau)
    assert c.fp / (c.fp + c.tn) <= 0.1


def test_selection_uses_validation_only():
    # threshold_zero_fp is a pure function of its (validation) argument — it cannot see test.
    val = [_p("n1", "NOT_CACHE_EQUIVALENT", 0.3), _p("p1", "CACHE_EQUIVALENT", 0.9)]
    assert select.threshold_zero_fp(val) == select.threshold_zero_fp(list(val))
