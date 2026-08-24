from sklearn.metrics import precision_score, recall_score

from polarity.eval import metrics
from polarity.types import Category, Label, Prediction

_DATA = [
    ("CACHE_EQUIVALENT", 0.98),
    ("CACHE_EQUIVALENT", 0.95),
    ("CACHE_EQUIVALENT", 0.91),
    ("CACHE_EQUIVALENT", 0.72),
    ("CACHE_EQUIVALENT", 0.55),
    ("CACHE_EQUIVALENT", 0.34),
    ("NOT_CACHE_EQUIVALENT", 0.10),
    ("NOT_CACHE_EQUIVALENT", 0.22),
    ("NOT_CACHE_EQUIVALENT", 0.48),
    ("NOT_CACHE_EQUIVALENT", 0.60),
    ("NOT_CACHE_EQUIVALENT", 0.88),
    ("NOT_CACHE_EQUIVALENT", 0.93),
]


def _preds() -> list[Prediction]:
    return [
        Prediction(
            pair_id=f"p{i}",
            gold=Label(g),
            score=s,
            category=Category.PARAPHRASE,
            source="t",
        )
        for i, (g, s) in enumerate(_DATA)
    ]


def test_confusion_hand_computed():
    c = metrics.confusion_at(_preds(), 0.5)
    assert (c.tp, c.fn, c.fp, c.tn) == (5, 1, 3, 3)


def test_false_positive_is_negative_predicted_positive():
    p = [
        Prediction(
            pair_id="n",
            gold=Label.NOT_CACHE_EQUIVALENT,
            score=0.9,
            category=Category.EXPLICIT_NEGATION,
            source="t",
        )
    ]
    assert metrics.confusion_at(p, 0.5).fp == 1
    assert metrics.confusion_at(p, 0.95).fp == 0


def test_recall_at_zero_fp():
    op = metrics.recall_at_zero_fp(_preds())
    assert op.fp == 0
    assert op.n_negatives == 6
    assert op.max_negative_score == 0.93
    assert op.recall == round(2 / 6, 6)  # only 0.98 and 0.95 clear the top negative


def test_matches_sklearn():
    preds = _preds()
    tau = 0.5
    row = metrics._row(preds, tau)
    y_true = [1 if p.gold == Label.CACHE_EQUIVALENT else 0 for p in preds]
    y_pred = [1 if p.score >= tau else 0 for p in preds]
    assert round(row.precision, 6) == round(float(precision_score(y_true, y_pred)), 6)
    assert round(row.recall, 6) == round(float(recall_score(y_true, y_pred)), 6)


def test_serialize_is_deterministic():
    preds = _preds()
    assert metrics.serialize_report(metrics.build_report(preds)) == metrics.serialize_report(
        metrics.build_report(preds)
    )
