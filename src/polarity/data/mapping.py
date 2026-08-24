"""Label-mapping bookkeeping.

Turns an upstream row into a canonical ``Example`` while keeping the derived cache-equivalence
label distinct from the retained upstream label, and records a ``MappingAudit`` — the counts
through every stage (source rows -> eligible -> mapped -> rejected, with reasons) that the M1
report reconciles.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from polarity.data.categorize import categorize
from polarity.data.schema import Example, RecipeTier, provenance_hash, text_shas
from polarity.types import Label


class MappingAudit(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    source_rows: int
    accepted: int
    rejected: int
    rejections: dict[str, int]  # reason -> count

    def reconciles(self) -> bool:
        return self.accepted + self.rejected == self.source_rows and self.rejected == sum(
            self.rejections.values()
        )


def make_example(
    *,
    source: str,
    revision: str,
    source_split: str,
    source_example_id: str,
    text_a: str,
    text_b: str,
    original_label: str,
    label: Label | None,
    mapping_rule: str,
    license_id: str,
    tier: RecipeTier,
) -> Example:
    """Build one canonical example; ``source_family_id`` is filled later by grouping."""
    sha_a, sha_b = text_shas(text_a, text_b)
    return Example(
        example_id=f"{source}:{source_split}:{source_example_id}",
        source_dataset=source,
        source_revision=revision,
        source_split=source_split,
        source_example_id=str(source_example_id),
        text_a=text_a,
        text_b=text_b,
        original_label=str(original_label),
        label=label,
        label_mapping_rule=mapping_rule,
        category=categorize(text_a, text_b, label),
        source_family_id="",
        transformation_parent_id=None,
        provenance_hash=provenance_hash(
            source_dataset=source,
            source_revision=revision,
            source_example_id=str(source_example_id),
            mapping_rule=mapping_rule,
        ),
        license_id=license_id,
        recipe_tier=tier,
        text_a_sha=sha_a,
        text_b_sha=sha_b,
    )


__all__ = ["MappingAudit", "make_example"]
