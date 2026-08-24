from collections import defaultdict

import pytest

from polarity.data.grouping import with_families
from polarity.data.schema import Example, RecipeTier
from polarity.data.split import assign_family_splits, transform_split_offenders
from polarity.types import Category, Label, SplitName


def _ex(eid: str, a: str, b: str, parent: str | None = None) -> Example:
    return Example(
        example_id=eid,
        source_dataset="s",
        source_revision="r",
        source_split="x",
        source_example_id=eid,
        text_a=a,
        text_b=b,
        original_label="1",
        label=Label.CACHE_EQUIVALENT,
        label_mapping_rule="m",
        category=Category.PARAPHRASE,
        source_family_id="",
        transformation_parent_id=parent,
        provenance_hash="p",
        license_id="l",
        recipe_tier=RecipeTier.PRIMARY,
        text_a_sha="sa",
        text_b_sha="sb",
    )


def _sample(n: int) -> list[Example]:
    return with_families([_ex(str(i), f"t{i}a", f"t{i}b") for i in range(n)])


def test_family_never_crosses_split():
    exs = _sample(40)
    split_of = assign_family_splits(exs)
    fam_split: dict[str, set[SplitName]] = defaultdict(set)
    for e in exs:
        fam_split[e.source_family_id].add(split_of[e.example_id])
    assert all(len(splits) == 1 for splits in fam_split.values())


def test_deterministic():
    exs = _sample(20)
    assert assign_family_splits(exs) == assign_family_splits(exs)


def test_ratios_must_sum_to_one():
    with pytest.raises(ValueError):
        assign_family_splits(_sample(3), train=0.5, validation=0.2, test=0.2)


def test_split_depends_only_on_family_so_removing_sources_is_safe():
    # A retained example's split is unchanged when other examples (e.g. MNLI) are removed —
    # the core recipe-separability guarantee: primary is a clean subset of primary+MNLI.
    exs = _sample(20)
    full = assign_family_splits(exs)
    subset = assign_family_splits(exs[:10])
    assert all(full[e.example_id] == subset[e.example_id] for e in exs[:10])


def test_transform_child_must_match_parent_split():
    exs = with_families([_ex("p", "pa", "pb"), _ex("c", "ca", "cb", parent="p")])
    split_of = assign_family_splits(exs)
    split_of["c"], split_of["p"] = SplitName.TRAIN, SplitName.TEST  # force a mismatch
    assert "c" in transform_split_offenders(exs, split_of)


def test_no_offenders_without_parents():
    exs = _sample(5)
    assert transform_split_offenders(exs, assign_family_splits(exs)) == []
