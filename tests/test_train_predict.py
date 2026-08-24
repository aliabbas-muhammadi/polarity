from pathlib import Path

from polarity.baselines.materialize import MaterializedPair
from polarity.baselines.schema import BaselinePrediction
from polarity.train import predict
from polarity.types import Category, Label


def _pair(pid: str, gold: str) -> MaterializedPair:
    return MaterializedPair(
        example_id=pid,
        text_a="the service is not available",
        text_b="the service is unavailable",
        gold=Label(gold),
        category=Category.EQUIVALENT_NEGATION,
        source="cannot",
    )


def test_label_index_convention():
    assert predict.POSITIVE_INDEX == 1
    assert predict.label_index(Label.CACHE_EQUIVALENT) == 1
    assert predict.label_index(Label.NOT_CACHE_EQUIVALENT) == 0


def test_to_prediction_rounds_and_fills_shortcut_features():
    p = predict.to_prediction(_pair("cannot:all:1", "CACHE_EQUIVALENT"), 0.1234567)
    assert isinstance(p, BaselinePrediction)
    assert p.score == 0.123457  # rounded to 6dp
    assert p.has_neg_cue is True  # "not" present
    assert 0.0 <= p.token_overlap <= 1.0


def test_write_predictions_schema_sorted_and_reloadable(tmp_path: Path):
    pairs = [
        _pair("cannot:all:2", "NOT_CACHE_EQUIVALENT"),
        _pair("cannot:all:1", "CACHE_EQUIVALENT"),
    ]
    scores = [0.9, 0.1]
    out = tmp_path / "validation.jsonl"
    predict.write_predictions(pairs, scores, out)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert out.read_text(encoding="utf-8").endswith("\n")
    reloaded = [BaselinePrediction.model_validate_json(x) for x in lines]
    assert [p.pair_id for p in reloaded] == ["cannot:all:1", "cannot:all:2"]  # sorted by pair_id
