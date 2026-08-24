from polarity.baselines.deterministic import score_exact, score_normalized


def test_exact():
    assert score_exact("a", "a") == 1.0
    assert score_exact("a", "b") == 0.0


def test_normalized_equates_only_surface_variants():
    assert score_normalized("Hello, World!", "hello world") == 1.0


def test_normalized_keeps_negation():
    # normalize must NOT strip "not"/antonyms, so a meaning change stays non-equal
    assert score_normalized("it is not safe", "it is safe") == 0.0
    assert score_normalized("15 apples", "16 apples") == 0.0
