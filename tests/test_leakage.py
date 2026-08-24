import pytest

from polarity.eval.leakage import LeakageError, assert_holdout_isolated, audit, normalize
from polarity.types import Category, Label, Pair, SplitManifestEntry, SplitName


def _p(pid: str, a: str, b: str, group: str = "g") -> Pair:
    return Pair(
        id=pid,
        group=group,
        text_a=a,
        text_b=b,
        label=Label.CACHE_EQUIVALENT,
        category=Category.PARAPHRASE,
        source="t",
    )


def test_clean_sets_ok():
    assert audit([_p("t1", "alpha", "beta")], [_p("h1", "gamma", "delta")]).ok


def test_detects_exact_overlap():
    report = audit([_p("t1", "shared text", "beta")], [_p("h1", "shared text", "delta")])
    assert "shared text" in report.exact_overlaps
    assert not report.ok


def test_detects_normalized_overlap():
    report = audit([_p("t1", "Hello, World!", "beta")], [_p("h1", "hello world", "delta")])
    assert report.normalized_overlaps
    assert not report.ok


def test_detects_reverse_pair():
    report = audit([_p("t1", "a", "b")], [_p("h1", "b", "a")])
    assert "h1" in report.reverse_pair_overlaps


def test_detects_cross_split_group():
    entries = [
        SplitManifestEntry(pair_id="1", source="s", group="g", split=SplitName.TRAIN, pair_sha="x"),
        SplitManifestEntry(pair_id="2", source="s", group="g", split=SplitName.TEST, pair_sha="y"),
    ]
    assert "g" in audit([], [], entries).cross_split_groups


def test_assert_isolated_fails_closed():
    with pytest.raises(LeakageError):
        assert_holdout_isolated([_p("t1", "secret q", "beta")], [_p("h1", "secret q", "delta")])


def test_normalize():
    assert normalize("  Hello, WORLD!! ") == "hello world"
