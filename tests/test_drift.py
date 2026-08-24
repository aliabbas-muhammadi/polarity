from pathlib import Path

from polarity.eval import metrics

ROOT = Path(__file__).resolve().parents[1]


def test_committed_report_reproduces_byte_for_byte():
    preds = metrics.load_predictions(ROOT / "predictions" / "synthetic.jsonl")
    fresh = metrics.serialize_report(metrics.build_report(preds))
    committed = (ROOT / "reports" / "metrics.json").read_text(encoding="utf-8")
    assert fresh == committed


def test_report_recomputation_is_stable():
    preds = metrics.load_predictions(ROOT / "predictions" / "synthetic.jsonl")
    first = metrics.serialize_report(metrics.build_report(preds))
    second = metrics.serialize_report(metrics.build_report(preds))
    assert first == second
