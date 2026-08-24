from polarity.data.grouping import assign_families, with_families
from polarity.data.schema import Example, RecipeTier
from polarity.types import Category, Label


def _ex(eid: str, a: str, b: str, source: str = "s") -> Example:
    return Example(
        example_id=eid,
        source_dataset=source,
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
        transformation_parent_id=None,
        provenance_hash="p",
        license_id="l",
        recipe_tier=RecipeTier.PRIMARY,
        text_a_sha="sa",
        text_b_sha="sb",
    )


def test_family_transitivity():
    exs = [_ex("1", "A", "B"), _ex("2", "A", "C"), _ex("3", "C", "D")]
    fams = assign_families(exs)
    assert fams["1"] == fams["2"] == fams["3"]


def test_disjoint_families_differ():
    fams = assign_families([_ex("1", "A", "B"), _ex("2", "X", "Y")])
    assert fams["1"] != fams["2"]


def test_deterministic_regardless_of_order():
    exs = [_ex("1", "A", "B"), _ex("2", "B", "C")]
    assert assign_families(exs) == assign_families(list(reversed(exs)))


def test_cross_dataset_family():
    exs = [_ex("1", "shared text", "B", source="paws"), _ex("2", "shared text", "C", source="mnli")]
    fams = assign_families(exs)
    assert fams["1"] == fams["2"]


def test_with_families_assigns_ids():
    out = with_families([_ex("1", "A", "B")])
    assert out[0].source_family_id.startswith("fam:")
