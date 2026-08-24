"""Source-family grouping via the sentence-pair graph.

Each example links its two normalized texts; connected components over ALL examples (across
datasets) form source families. Every member of a family must land in one split, so a source
sentence cannot leak across train/val/test through a variant — even a variant that lives in a
different dataset. Transitive by construction (A-B, A-C, C-D collapse to one family).
Deterministic: family id = ``fam:`` + short hash of the component's lexicographically smallest
normalized text, independent of iteration/union order.
"""

from __future__ import annotations

import hashlib

from polarity.data.schema import Example
from polarity.textnorm import normalize


def assign_families(examples: list[Example]) -> dict[str, str]:
    """Return ``example_id -> source_family_id`` via union-find over shared normalized text."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            lo, hi = (ra, rb) if ra < rb else (rb, ra)
            parent[hi] = lo

    for e in examples:
        na, nb = normalize(e.text_a), normalize(e.text_b)
        find(na)
        find(nb)
        union(na, nb)

    component_min: dict[str, str] = {}
    for text in parent:
        root = find(text)
        if root not in component_min or text < component_min[root]:
            component_min[root] = text

    def family_id(text: str) -> str:
        digest = hashlib.sha256(component_min[find(text)].encode("utf-8")).hexdigest()[:16]
        return f"fam:{digest}"

    return {e.example_id: family_id(normalize(e.text_a)) for e in examples}


def with_families(examples: list[Example]) -> list[Example]:
    """Return copies of ``examples`` with ``source_family_id`` assigned."""
    families = assign_families(examples)
    return [e.model_copy(update={"source_family_id": families[e.example_id]}) for e in examples]


__all__ = ["assign_families", "with_families"]
