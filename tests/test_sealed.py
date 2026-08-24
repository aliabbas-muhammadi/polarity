import pytest

from polarity.data.schema import Example, RecipeTier
from polarity.eval.leakage import SealedLeakageError, assert_sealed_isolated, sealed_set_audit
from polarity.types import Category, Label


def _ex(eid: str, a: str, b: str) -> Example:
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
        source_family_id="fam:x",
        transformation_parent_id=None,
        provenance_hash="p",
        license_id="l",
        recipe_tier=RecipeTier.PRIMARY,
        text_a_sha="sa",
        text_b_sha="sb",
    )


def test_clean_isolated():
    assert sealed_set_audit([_ex("1", "train q", "other")], [("veritas base", "veritas probe")]).ok


def test_text_collision_detected():
    report = sealed_set_audit(
        [_ex("1", "Capital of France?", "para")], [("capital of france", "z")]
    )
    assert "1" in report.text_collisions
    assert not report.ok


def test_reverse_pair_detected():
    report = sealed_set_audit([_ex("1", "a", "b")], [("b", "a")])
    assert "1" in report.reverse_pair_collisions


def test_assert_fails_closed():
    with pytest.raises(SealedLeakageError):
        assert_sealed_isolated([_ex("1", "shared", "x")], [("shared", "y")])
