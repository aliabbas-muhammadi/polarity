"""Deterministic cross-dataset duplicate / overlap detection (no embeddings).

Transparent lexical checks only — exact/normalized text shared across sources, duplicated
normalized pairs, reversed pairs, and duplicate example ids. Embedding-based fuzzy dedup would
introduce another model + threshold before training, so it is deliberately out of M1.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from pydantic import BaseModel, ConfigDict

from polarity.data.schema import Example
from polarity.textnorm import normalize


class OverlapReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n_examples: int
    cross_source_shared_texts: int  # a normalized text appearing in >1 source dataset
    duplicate_normalized_pairs: int  # the same (norm a, norm b) appearing more than once
    reverse_pairs: int  # (a,b) present and (b,a) also present (a != b), counted once
    duplicate_example_ids: int

    @property
    def clean(self) -> bool:
        return self.duplicate_example_ids == 0


def detect(examples: list[Example]) -> OverlapReport:
    text_sources: dict[str, set[str]] = defaultdict(set)
    pair_counts: Counter[tuple[str, str]] = Counter()
    pairs: set[tuple[str, str]] = set()
    id_counts: Counter[str] = Counter(e.example_id for e in examples)

    for e in examples:
        a, b = normalize(e.text_a), normalize(e.text_b)
        text_sources[a].add(e.source_dataset)
        text_sources[b].add(e.source_dataset)
        pair_counts[(a, b)] += 1
        pairs.add((a, b))

    return OverlapReport(
        n_examples=len(examples),
        cross_source_shared_texts=sum(1 for sources in text_sources.values() if len(sources) > 1),
        duplicate_normalized_pairs=sum(1 for c in pair_counts.values() if c > 1),
        reverse_pairs=sum(1 for a, b in pairs if a < b and (b, a) in pairs),
        duplicate_example_ids=sum(1 for c in id_counts.values() if c > 1),
    )


__all__ = ["OverlapReport", "detect"]
