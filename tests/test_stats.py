from polarity.data import stats
from polarity.data.grouping import with_families
from polarity.data.schema import Example, RecipeTier
from polarity.types import Category, Label


def _ex(
    eid: str,
    a: str,
    b: str,
    label: Label,
    source: str,
    category: Category = Category.PARAPHRASE,
) -> Example:
    return Example(
        example_id=eid,
        source_dataset=source,
        source_revision="r",
        source_split="x",
        source_example_id=eid,
        text_a=a,
        text_b=b,
        original_label="1",
        label=label,
        label_mapping_rule="m",
        category=category,
        source_family_id="",
        transformation_parent_id=None,
        provenance_hash="p",
        license_id="l",
        recipe_tier=RecipeTier.PRIMARY,
        text_a_sha="sa",
        text_b_sha="sb",
    )


def test_counts_and_uninformative_source():
    # each source contributes both classes -> source does not determine the label
    exs = with_families(
        [
            _ex("1", "a1", "b1", Label.CACHE_EQUIVALENT, "paws"),
            _ex("2", "a2", "b2", Label.NOT_CACHE_EQUIVALENT, "paws"),
            _ex("3", "a3", "b3", Label.CACHE_EQUIVALENT, "cannot"),
            _ex("4", "a4", "b4", Label.NOT_CACHE_EQUIVALENT, "cannot"),
        ]
    )
    s = stats.compute(exs, "primary")
    assert s.n_examples == 4
    assert s.label_equivalent == 2
    assert s.label_not_equivalent == 2
    assert s.source_majority_accuracy == 0.5


def test_source_confound_flagged():
    # paws all-positive, mnli all-negative -> source determines label
    exs = with_families(
        [
            _ex("1", "a1", "b1", Label.CACHE_EQUIVALENT, "paws"),
            _ex("2", "a2", "b2", Label.CACHE_EQUIVALENT, "paws"),
            _ex("3", "a3", "b3", Label.NOT_CACHE_EQUIVALENT, "mnli"),
            _ex("4", "a4", "b4", Label.NOT_CACHE_EQUIVALENT, "mnli"),
        ]
    )
    s = stats.compute(exs, "with_mnli")
    assert s.source_majority_accuracy == 1.0
    assert any("source_predicts_label" in f for f in s.shortcut_flags)


def test_equivalent_negation_insufficient_flag():
    exs = with_families(
        [
            _ex(
                "1",
                "not secure",
                "insecure",
                Label.CACHE_EQUIVALENT,
                "cannot",
                category=Category.EQUIVALENT_NEGATION,
            )
        ]
    )
    s = stats.compute(exs, "primary")
    assert s.equivalent_negation_count == 1
    assert not s.equivalent_negation_sufficient
