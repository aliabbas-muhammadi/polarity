import pytest
from pydantic import ValidationError

from polarity.types import Category, Label, Pair, Prediction


def test_pair_roundtrip():
    p = Pair(
        id="x",
        group="g",
        text_a="a",
        text_b="b",
        label=Label.CACHE_EQUIVALENT,
        category=Category.PARAPHRASE,
        source="s",
    )
    assert Pair.model_validate_json(p.model_dump_json()) == p


def test_pair_forbids_extra_fields():
    with pytest.raises(ValidationError):
        Pair(
            id="x",
            group="g",
            text_a="a",
            text_b="b",
            label=Label.CACHE_EQUIVALENT,
            category=Category.PARAPHRASE,
            source="s",
            bogus=1,
        )


def test_prediction_score_bounds():
    with pytest.raises(ValidationError):
        Prediction(
            pair_id="p",
            gold=Label.CACHE_EQUIVALENT,
            score=1.5,
            category=Category.PARAPHRASE,
            source="s",
        )
    ok = Prediction(
        pair_id="p",
        gold=Label.CACHE_EQUIVALENT,
        score=1.0,
        category=Category.PARAPHRASE,
        source="s",
    )
    assert ok.score == 1.0


def test_labels_are_strings():
    assert Label.CACHE_EQUIVALENT == "CACHE_EQUIVALENT"
    assert Label.NOT_CACHE_EQUIVALENT != Label.CACHE_EQUIVALENT
