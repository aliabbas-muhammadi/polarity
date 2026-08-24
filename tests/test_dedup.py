import json
from pathlib import Path

from polarity.data.dedup import detect
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
        source_family_id="fam:x",
        transformation_parent_id=None,
        provenance_hash="p",
        license_id="l",
        recipe_tier=RecipeTier.PRIMARY,
        text_a_sha="sa",
        text_b_sha="sb",
    )


def test_cross_source_shared_text():
    report = detect(
        [_ex("1", "hello world", "x", source="paws"), _ex("2", "hello world", "y", source="mnli")]
    )
    assert report.cross_source_shared_texts >= 1


def test_reverse_pairs():
    assert detect([_ex("1", "a", "b"), _ex("2", "b", "a")]).reverse_pairs == 1


def test_duplicate_example_ids():
    report = detect([_ex("dup", "a", "b"), _ex("dup", "c", "d")])
    assert report.duplicate_example_ids == 1
    assert not report.clean


def test_clean_set():
    assert detect([_ex("1", "a", "b"), _ex("2", "c", "d")]).clean


def test_overlap_report_contains_counts_only_not_sentence_or_pair_samples():
    report = detect(
        [
            _ex("1", "synthetic alpha", "synthetic beta", source="source-a"),
            _ex("2", "synthetic alpha", "synthetic gamma", source="source-b"),
        ]
    ).model_dump(mode="json")

    assert report == {
        "n_examples": 2,
        "cross_source_shared_texts": 1,
        "duplicate_normalized_pairs": 0,
        "reverse_pairs": 0,
        "duplicate_example_ids": 0,
    }
    assert all(isinstance(value, int) for value in report.values())


def test_tracked_aggregate_report_content_inventory_has_no_sentence_or_pair_fields():
    forbidden_fields = {
        "base",
        "probe",
        "sentence1",
        "sentence2",
        "text_a",
        "text_b",
        "sample_shared_texts",
        "sample_reverse_pairs",
    }
    findings: list[str] = []

    def inventory(value: object, *, path: Path, location: str = "$") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in forbidden_fields:
                    findings.append(f"{path}:{location}.{key}")
                inventory(child, path=path, location=f"{location}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                inventory(child, path=path, location=f"{location}[{index}]")

    for path in sorted(Path("reports").rglob("*.json")):
        inventory(json.loads(path.read_text(encoding="utf-8")), path=path)

    assert findings == []


def test_m1_report_duplicate_pair_claim_matches_aggregate_report():
    aggregate = json.loads(Path("reports/m1/dedup.json").read_text(encoding="utf-8"))
    report = Path("docs/M1-REPORT.md").read_text(encoding="utf-8")

    assert f"duplicate normalized pairs **{aggregate['duplicate_normalized_pairs']}**" in report
