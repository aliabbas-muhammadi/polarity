"""Core typed records for Polarity.

Every artifact — a data pair, a split-manifest row, a model prediction — is a frozen
pydantic model, so a schema violation fails loudly and an artifact's identity is stable.
Nothing here knows about a model or a GPU; these are the contracts the M0 foundation rests on.

Label convention (see docs/SPEC.md): the POSITIVE class is CACHE_EQUIVALENT. This choice is
load-bearing for the false-positive definition in docs/METHODOLOGY.md and eval/metrics.py.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Label(StrEnum):
    """Binary cache-equivalence label. Positive = CACHE_EQUIVALENT."""

    CACHE_EQUIVALENT = "CACHE_EQUIVALENT"
    NOT_CACHE_EQUIVALENT = "NOT_CACHE_EQUIVALENT"


class Category(StrEnum):
    """Pre-registered semantic categories (docs/SPEC.md).

    A pair's category is the *kind* of relation between its two texts; it is independent of
    the label. No single aggregate metric may hide a per-category failure, so every pair
    carries one. ``MEANING_CHANGE_OTHER`` is a deliberately coarse umbrella for externally
    bucketed sets (e.g. the Veritas ``different-answer`` pairs); Polarity's own data uses the
    fine categories.
    """

    EXACT_REPEAT = "exact_repeat"
    PARAPHRASE = "paraphrase"
    HARD_PARAPHRASE = "hard_paraphrase"
    EQUIVALENT_NEGATION = "equivalent_negation"
    EXPLICIT_NEGATION = "explicit_negation"
    LEXICAL_CONTRADICTION = "lexical_contradiction"
    NUMERIC_CHANGE = "numeric_change"
    ENTITY_CHANGE = "entity_change"
    TEMPORAL_CHANGE = "temporal_change"
    QUANTIFIER_CHANGE = "quantifier_change"
    MODALITY_CHANGE = "modality_change"
    SCOPE_CHANGE = "scope_change"
    RELATION_REVERSAL = "relation_reversal"
    MEANING_CHANGE_OTHER = "meaning_change_other"
    UNKNOWN = "unknown"  # evidence insufficient to categorize; reported, never forced


class SplitName(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    FINAL_HOLDOUT = "final_holdout"


class Pair(BaseModel):
    """A single labelled sentence pair.

    ``group`` is the *source-family* key: every augmented variant of one source sentence
    shares it, and splitting happens over groups so a source can never appear in two splits
    through different surface forms (docs/METHODOLOGY.md).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    group: str
    text_a: str
    text_b: str
    label: Label
    category: Category
    source: str
    note: str = ""


class SplitManifestEntry(BaseModel):
    """One immutable row of a committed split manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pair_id: str
    source: str
    group: str
    split: SplitName
    pair_sha: str  # content hash of (text_a, text_b, label); detects silent edits


class Prediction(BaseModel):
    """A model's score for a pair. ``score`` = P(CACHE_EQUIVALENT)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pair_id: str
    gold: Label
    score: float = Field(ge=0.0, le=1.0)
    category: Category
    source: str


__all__ = ["Category", "Label", "Pair", "Prediction", "SplitManifestEntry", "SplitName"]
