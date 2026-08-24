"""Loaders for committed data artifacts: synthetic fixtures and the sealed Veritas snapshot.

The snapshot is stored faithfully in its source (Veritas) schema; ``golden_to_pairs`` maps
it into Polarity ``Pair`` records only for the leakage audit. That pairType -> Category
mapping is deliberately coarse and never feeds a metric in M0 (see the function docstring).
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from polarity.types import Category, Label, Pair


class GoldenPair(BaseModel):
    """One pair from the sealed Veritas golden-set snapshot (source schema, verbatim)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    group: str
    base: str
    probe: str
    pairType: str
    equivalent: bool
    note: str = ""


_PAIRTYPE_TO_CATEGORY = {
    "exact-repeat": Category.EXACT_REPEAT,
    "paraphrase-should-hit": Category.PARAPHRASE,
    "negation-must-not-hit": Category.EXPLICIT_NEGATION,
    "different-answer-must-not-hit": Category.MEANING_CHANGE_OTHER,
}


def load_pairs(path: Path) -> list[Pair]:
    """Load synthetic fixtures (one JSON ``Pair`` per line)."""
    out: list[Pair] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line:
            out.append(Pair.model_validate_json(line))
    return out


def load_golden(path: Path) -> list[GoldenPair]:
    """Load the sealed Veritas snapshot (a JSON array in the source schema)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [GoldenPair.model_validate(row) for row in data]


def golden_to_pairs(golden: list[GoldenPair]) -> list[Pair]:
    """Map snapshot rows into ``Pair`` records for the leakage/isolation audit.

    The pairType -> Category mapping is coarse (Veritas uses a 4-way taxonomy;
    'different-answer' collapses to ``MEANING_CHANGE_OTHER``). This mapping feeds only the
    text-overlap isolation check in M0 — never a metric — so the coarseness is harmless here.
    """
    return [
        Pair(
            id=g.id,
            group=g.group,
            text_a=g.base,
            text_b=g.probe,
            label=Label.CACHE_EQUIVALENT if g.equivalent else Label.NOT_CACHE_EQUIVALENT,
            category=_PAIRTYPE_TO_CATEGORY.get(g.pairType, Category.MEANING_CHANGE_OTHER),
            source="veritas-golden",
            note=g.note,
        )
        for g in golden
    ]


__all__ = ["GoldenPair", "golden_to_pairs", "load_golden", "load_pairs"]
