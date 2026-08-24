import pytest

from polarity.data.split import assign_splits, pair_sha, union_find_groups
from polarity.types import Category, Label, Pair, SplitName


def _p(pid: str, group: str, a: str, b: str) -> Pair:
    return Pair(
        id=pid,
        group=group,
        text_a=a,
        text_b=b,
        label=Label.CACHE_EQUIVALENT,
        category=Category.PARAPHRASE,
        source="t",
    )


def _sample() -> list[Pair]:
    return [
        _p("1", "g1", "a1", "b1"),
        _p("2", "g1", "a2", "b2"),  # shares group g1 with pair 1
        _p("3", "g2", "a3", "b3"),
        _p("4", "g3", "a4", "b4"),
        _p("5", "g4", "a5", "b5"),
        _p("6", "g5", "a6", "b6"),
    ]


def test_deterministic_same_seed():
    assert assign_splits(_sample(), seed=42) == assign_splits(_sample(), seed=42)


def test_group_never_spans_two_splits():
    by_group: dict[str, set[SplitName]] = {}
    for e in assign_splits(_sample(), seed=42):
        by_group.setdefault(e.group, set()).add(e.split)
    assert all(len(splits) == 1 for splits in by_group.values())


def test_ratios_must_sum_to_one():
    with pytest.raises(ValueError):
        assign_splits(_sample(), train=0.5, validation=0.2, test=0.2)


def test_pinned_group_forced():
    entries = assign_splits(_sample(), pinned={"g2": SplitName.FINAL_HOLDOUT})
    assert all(e.split == SplitName.FINAL_HOLDOUT for e in entries if e.group == "g2")


def test_union_find_merges_shared_text():
    pairs = [_p("1", "gA", "x", "y"), _p("2", "gB", "y", "z")]  # linked through "y"
    groups = union_find_groups(pairs)
    assert groups["1"] == groups["2"]


def test_pair_sha_sensitive_to_text():
    assert pair_sha(_p("1", "g", "a", "b")) != pair_sha(_p("1", "g", "a", "c"))
