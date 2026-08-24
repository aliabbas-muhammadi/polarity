import pytest
from sklearn.metrics import brier_score_loss

from polarity.calibration import brier_score, expected_calibration_error


def test_brier_hand_computed():
    assert brier_score([1.0, 0.0], [1, 0]) == 0.0
    assert brier_score([0.0, 1.0], [1, 0]) == 1.0


def test_brier_matches_sklearn():
    probs = [0.9, 0.1, 0.8, 0.3]
    labels = [1, 0, 1, 0]
    assert round(brier_score(probs, labels), 9) == round(float(brier_score_loss(labels, probs)), 9)


def test_ece_perfectly_calibrated_is_zero():
    assert expected_calibration_error([0.0, 1.0], [0, 1], n_bins=10) == 0.0


def test_ece_confident_and_wrong_is_one():
    assert expected_calibration_error([1.0, 1.0], [0, 0], n_bins=10) == 1.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        brier_score([0.5], [0, 1])
