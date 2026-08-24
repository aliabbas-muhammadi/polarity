import pytest
from pydantic import ValidationError

from polarity.baselines.cross_encoder import symmetric_min
from polarity.baselines.schema import BaselinePrediction, shortcut_features
from polarity.types import Category, Label, Prediction


def _bp(score: float) -> BaselinePrediction:
    return BaselinePrediction(
        pair_id="x",
        gold=Label.CACHE_EQUIVALENT,
        score=score,
        category=Category.PARAPHRASE,
        source="s",
        has_neg_cue=False,
        token_overlap=0.5,
        len_diff=0,
    )


def test_shortcut_features_detect_negation_cue():
    cue, overlap, len_diff = shortcut_features("Is it not safe?", "Is it safe?")
    assert cue is True
    assert 0.0 <= overlap <= 1.0
    assert len_diff >= 0


def test_core_conversion():
    core = _bp(0.5).core()
    assert isinstance(core, Prediction)
    assert core.score == 0.5


def test_score_bounds():
    with pytest.raises(ValidationError):
        _bp(1.5)


def test_symmetric_min_is_strict_and_symmetric():
    assert symmetric_min([0.9, 0.2], [0.3, 0.8]) == [0.3, 0.2]
