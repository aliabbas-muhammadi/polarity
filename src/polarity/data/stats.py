"""Balance, coverage, and shortcut-risk analysis over a set of examples.

Answers the questions M1 must answer before any recipe is called ready: what is the label /
category / source balance, how much equivalent-negation coverage exists, and — most important —
could a model predict the label from DATASET STYLE instead of semantics? The confound headline
``source_majority_accuracy`` is the accuracy of always guessing a source's majority class; near
the base rate ⇒ source is uninformative, near 1.0 ⇒ source ~determines the label.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from pydantic import BaseModel, ConfigDict

from polarity.data.categorize import NEGATION_CUES
from polarity.data.schema import Example
from polarity.textnorm import normalize
from polarity.types import Category, Label

_R = 6
EQUIVALENT_NEGATION_MIN = 200  # coverage floor below which we flag INSUFFICIENT


def _r(x: float) -> float:
    return round(x, _R)


def _pctile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))]


class SourceContribution(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    equivalent: int
    not_equivalent: int
    unlabeled: int
    total: int
    p_not_given_source: float


class CategoryCount(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    category: str
    n: int


class RecipeStats(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    recipe: str
    n_examples: int
    n_families: int
    family_size_p50: int
    family_size_p95: int
    family_size_max: int
    label_equivalent: int
    label_not_equivalent: int
    label_unlabeled: int
    per_source: list[SourceContribution]
    categories: list[CategoryCount]
    equivalent_negation_count: int
    equivalent_negation_sufficient: bool
    char_len_p50: int
    char_len_p95: int
    token_len_p50: int
    token_len_p95: int
    frac_equivalent_with_negation_cue: float
    frac_not_with_negation_cue: float
    source_majority_accuracy: float  # confound headline (0.5 base .. 1.0 = source determines label)
    shortcut_flags: list[str]


def _has_neg_cue(text: str) -> bool:
    return bool(NEGATION_CUES & set(normalize(text).split()))


def compute(examples: list[Example], recipe: str) -> RecipeStats:
    n = len(examples)
    fam_members: dict[str, int] = Counter(e.source_family_id for e in examples)
    labels = Counter(e.label for e in examples)
    n_eq = labels.get(Label.CACHE_EQUIVALENT, 0)
    n_not = labels.get(Label.NOT_CACHE_EQUIVALENT, 0)
    n_unl = labels.get(None, 0)

    by_source: dict[str, Counter[Label | None]] = defaultdict(Counter)
    for e in examples:
        by_source[e.source_dataset][e.label] += 1
    per_source = [
        SourceContribution(
            source=src,
            equivalent=c.get(Label.CACHE_EQUIVALENT, 0),
            not_equivalent=c.get(Label.NOT_CACHE_EQUIVALENT, 0),
            unlabeled=c.get(None, 0),
            total=sum(c.values()),
            p_not_given_source=_r(c.get(Label.NOT_CACHE_EQUIVALENT, 0) / sum(c.values()))
            if sum(c.values())
            else 0.0,
        )
        for src, c in sorted(by_source.items())
    ]

    # Confound headline: accuracy of predicting each source's majority label.
    labeled = [(e.source_dataset, e.label) for e in examples if e.label is not None]
    correct = 0
    for counts in by_source.values():
        correct += max(
            counts.get(Label.CACHE_EQUIVALENT, 0), counts.get(Label.NOT_CACHE_EQUIVALENT, 0)
        )
    source_majority_acc = _r(correct / len(labeled)) if labeled else 0.0

    cats = Counter(e.category for e in examples)
    categories = [
        CategoryCount(category=cat.value, n=cats.get(cat, 0))
        for cat in Category
        if cats.get(cat, 0)
    ]
    eq_neg = cats.get(Category.EQUIVALENT_NEGATION, 0)

    char_lens = [len(e.text_a) + len(e.text_b) for e in examples]
    tok_lens = [len(e.text_a.split()) + len(e.text_b.split()) for e in examples]

    eq_examples = [e for e in examples if e.label == Label.CACHE_EQUIVALENT]
    not_examples = [e for e in examples if e.label == Label.NOT_CACHE_EQUIVALENT]
    frac_eq_neg = (
        _r(
            sum(1 for e in eq_examples if _has_neg_cue(e.text_a) or _has_neg_cue(e.text_b))
            / len(eq_examples)
        )
        if eq_examples
        else 0.0
    )
    frac_not_neg = (
        _r(
            sum(1 for e in not_examples if _has_neg_cue(e.text_a) or _has_neg_cue(e.text_b))
            / len(not_examples)
        )
        if not_examples
        else 0.0
    )

    flags: list[str] = []
    if source_majority_acc > 0.8:
        flags.append(f"source_predicts_label (source_majority_accuracy={source_majority_acc})")
    if frac_not_neg - frac_eq_neg > 0.6:
        flags.append(f"negation_token_shortcut (not_cue={frac_not_neg} vs equiv_cue={frac_eq_neg})")
    for sc in per_source:
        if n_not and sc.not_equivalent / n_not > 0.9 and len(per_source) > 1:
            flags.append(f"source_class_monopoly ({sc.source} supplies >90% of negatives)")
    if eq_neg < EQUIVALENT_NEGATION_MIN:
        flags.append(
            f"EQUIVALENT_NEGATION_COVERAGE_INSUFFICIENT (n={eq_neg} < {EQUIVALENT_NEGATION_MIN})"
        )

    sizes = list(fam_members.values())
    return RecipeStats(
        recipe=recipe,
        n_examples=n,
        n_families=len(fam_members),
        family_size_p50=_pctile(sizes, 0.5),
        family_size_p95=_pctile(sizes, 0.95),
        family_size_max=max(sizes) if sizes else 0,
        label_equivalent=n_eq,
        label_not_equivalent=n_not,
        label_unlabeled=n_unl,
        per_source=per_source,
        categories=categories,
        equivalent_negation_count=eq_neg,
        equivalent_negation_sufficient=eq_neg >= EQUIVALENT_NEGATION_MIN,
        char_len_p50=_pctile(char_lens, 0.5),
        char_len_p95=_pctile(char_lens, 0.95),
        token_len_p50=_pctile(tok_lens, 0.5),
        token_len_p95=_pctile(tok_lens, 0.95),
        frac_equivalent_with_negation_cue=frac_eq_neg,
        frac_not_with_negation_cue=frac_not_neg,
        source_majority_accuracy=source_majority_acc,
        shortcut_flags=flags,
    )


__all__ = [
    "EQUIVALENT_NEGATION_MIN",
    "CategoryCount",
    "RecipeStats",
    "SourceContribution",
    "compute",
]
