from polarity.data.mapping import MappingAudit, make_example
from polarity.data.schema import RecipeTier
from polarity.types import Label


def test_audit_reconciles():
    audit = MappingAudit(
        source="s", source_rows=10, accepted=7, rejected=3, rejections={"x": 2, "y": 1}
    )
    assert audit.reconciles()


def test_audit_does_not_reconcile_on_mismatch():
    audit = MappingAudit(source="s", source_rows=10, accepted=7, rejected=3, rejections={"x": 1})
    assert not audit.reconciles()


def test_make_example_carries_provenance_and_distinct_labels():
    e = make_example(
        source="paws",
        revision="r",
        source_split="train",
        source_example_id="1",
        text_a="A",
        text_b="B",
        original_label="1",
        label=Label.CACHE_EQUIVALENT,
        mapping_rule="paws:label1->equivalent",
        license_id="paws",
        tier=RecipeTier.PRIMARY,
    )
    assert e.example_id == "paws:train:1"
    assert e.provenance_hash.startswith("pl1:")
    assert e.text_a_sha and e.text_b_sha
    assert e.original_label == "1"
    assert e.label == Label.CACHE_EQUIVALENT
