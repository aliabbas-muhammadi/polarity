from polarity.calibration import fit


def _miscalibrated():
    # underconfident-then-overconfident synthetic set with a clear signal
    probs = [0.2, 0.25, 0.3, 0.35, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85] * 8
    labels = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1] * 8
    return probs, labels


def test_temperature_and_platt_reduce_nll():
    probs, labels = _miscalibrated()
    raw = fit.nll(probs, labels)
    for cal in (fit.fit_temperature(probs, labels), fit.fit_platt(probs, labels)):
        assert fit.nll(cal.apply(probs), labels) <= raw + 1e-9


def test_temperature_is_monotonic_preserving_ranking():
    probs, labels = _miscalibrated()
    cal = fit.fit_temperature(probs, labels)
    out = cal.apply([0.3, 0.5, 0.7])
    assert out[0] < out[1] < out[2]  # ranking (hence the FP/recall operating point) is preserved


def test_isotonic_none_below_min_obs():
    probs, labels = [0.2, 0.8], [0, 1]
    assert fit.fit_isotonic(probs, labels, min_obs=50) is None


def test_isotonic_reduces_nll_with_enough_data():
    probs, labels = _miscalibrated()
    iso = fit.fit_isotonic(probs, labels, min_obs=50)
    assert iso is not None
    assert fit.nll(iso.apply(probs), labels) <= fit.nll(probs, labels) + 1e-9


def test_calibrator_apply_clips_to_open_interval():
    cal = fit.identity()
    out = cal.apply([0.0, 1.0])
    assert out[0] > 0.0 and out[1] < 1.0


def test_fit_ignores_data_it_was_not_given():
    # fitting on validation only: the fit depends solely on the passed arrays
    probs, labels = _miscalibrated()
    a = fit.fit_temperature(probs, labels).params
    b = fit.fit_temperature([*probs, 0.99], [*labels, 0]).params  # extra point changes the fit
    assert a != b
