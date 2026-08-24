"""The canonical intermediate representation for M1.

Heterogeneous sources (PAWS-Wiki TSV, CANNOT TSV, MNLI JSONL, NaN-NLI CSV) are normalized into
one ``Example`` that retains enough provenance to answer, per row: where it came from, what its
upstream label was, why it now carries a cache-equivalence label, what semantic phenomenon it
tests, and which source-family / recipe tier it belongs to. The derived cache-equivalence
``label`` is kept strictly distinct from the retained ``original_label``.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from polarity.fingerprint import content_hash
from polarity.textnorm import normalized_sha
from polarity.types import Category, Label


class RecipeTier(StrEnum):
    """Which recipe an example belongs to. Kept separable end-to-end (MNLI is exploratory)."""

    PRIMARY = "primary"  # PAWS-Wiki + CANNOT
    EXPLORATORY_MNLI = "exploratory_mnli"  # + MNLI, never merged irreversibly
    EVAL_PROBE = "eval_probe"  # NaN-NLI etc.; never in a training manifest


class Example(BaseModel):
    """One normalized, provenance-carrying pair."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    example_id: str
    source_dataset: str
    source_revision: str
    source_split: str
    source_example_id: str
    text_a: str
    text_b: str
    original_label: str  # verbatim upstream label — never overwritten by the mapping
    label: Label | None  # derived cache-equivalence label; None for unmapped eval probes
    label_mapping_rule: str
    category: Category
    source_family_id: str
    transformation_parent_id: str | None
    provenance_hash: str
    license_id: str
    recipe_tier: RecipeTier
    text_a_sha: str  # sha256 of normalized text_a — text-free identity for leakage checks
    text_b_sha: str


def provenance_hash(
    *, source_dataset: str, source_revision: str, source_example_id: str, mapping_rule: str
) -> str:
    """Content-addressed identity of an example's origin + mapping."""
    return content_hash(
        {
            "dataset": source_dataset,
            "revision": source_revision,
            "source_example_id": source_example_id,
            "mapping_rule": mapping_rule,
        }
    )


def text_shas(text_a: str, text_b: str) -> tuple[str, str]:
    return normalized_sha(text_a), normalized_sha(text_b)


__all__ = ["Example", "RecipeTier", "provenance_hash", "text_shas"]
