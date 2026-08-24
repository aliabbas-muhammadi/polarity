"""Deterministic, leakage-safe splitting.

Splitting happens over GROUPS (source families), never individual pairs: every pair that
shares a ``group`` — all augmented variants of one source sentence — lands in the same
split, so a sentence can never appear in both train and eval through a different
transformation. A group's split is a pure function of ``(seed, group)``, independent of
dataset order, so the manifest is reproducible.
"""

from __future__ import annotations

import hashlib

from polarity.data.schema import Example
from polarity.fingerprint import content_hash
from polarity.types import Pair, SplitManifestEntry, SplitName


def pair_sha(p: Pair) -> str:
    """Content hash of the result-affecting fields of a pair (text + label)."""
    return content_hash({"a": p.text_a, "b": p.text_b, "label": p.label.value})


def _bucket(group: str, seed: int) -> float:
    """Map a group id to a stable pseudo-uniform value in [0, 1)."""
    digest = hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()
    return int(digest[:16], 16) / float(1 << 64)


def union_find_groups(pairs: list[Pair]) -> dict[str, str]:
    """Group pair ids into connected components over shared TEXT nodes.

    A pair links its two texts; pairs sharing any text join the same component. Returns
    ``pair_id -> canonical group id`` (the lexicographically smallest text in the
    component). Used by the leakage audit to detect declared ``group`` labels that actually
    belong to the same source family.
    """
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
        if ra == rb:
            return
        hi, lo = (ra, rb) if ra > rb else (rb, ra)
        parent[hi] = lo

    for p in pairs:
        find(p.text_a)
        find(p.text_b)
        union(p.text_a, p.text_b)

    return {p.id: find(p.text_a) for p in pairs}


def assign_splits(
    pairs: list[Pair],
    *,
    train: float = 0.7,
    validation: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
    pinned: dict[str, SplitName] | None = None,
) -> list[SplitManifestEntry]:
    """Assign every pair to a split by its GROUP, deterministically.

    Ratios are over groups, not pairs. ``pinned`` forces named groups to a split (e.g. a
    group reserved as ``FINAL_HOLDOUT`` is never auto-assigned). Entries are returned sorted
    by ``pair_id`` for a stable manifest.
    """
    total = train + validation + test
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")
    pinned = pinned or {}

    assignment: dict[str, SplitName] = {}
    for group in sorted({p.group for p in pairs}):
        if group in pinned:
            assignment[group] = pinned[group]
            continue
        b = _bucket(group, seed)
        if b < train:
            assignment[group] = SplitName.TRAIN
        elif b < train + validation:
            assignment[group] = SplitName.VALIDATION
        else:
            assignment[group] = SplitName.TEST

    return [
        SplitManifestEntry(
            pair_id=p.id,
            source=p.source,
            group=p.group,
            split=assignment[p.group],
            pair_sha=pair_sha(p),
        )
        for p in sorted(pairs, key=lambda x: x.id)
    ]


def _family_bucket(family_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{family_id}".encode()).hexdigest()
    return int(digest[:16], 16) / float(1 << 64)


def assign_family_splits(
    examples: list[Example],
    *,
    train: float = 0.8,
    validation: float = 0.1,
    test: float = 0.1,
    seed: int = 42,
) -> dict[str, SplitName]:
    """Assign each example to train/validation/test by its ``source_family_id``.

    The split is a pure function of ``(seed, family_id)``, so all members of a family share a
    split AND a family's split is identical whether or not other sources (e.g. MNLI) are
    present — making a primary-only recipe a clean subset of a primary+MNLI recipe. Veritas is
    never in this 3-way (it is the separate FINAL_HOLDOUT).
    """
    total = train + validation + test
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")
    out: dict[str, SplitName] = {}
    for e in examples:
        if not e.source_family_id:
            raise ValueError(f"{e.example_id}: no source_family_id (run grouping first)")
        b = _family_bucket(e.source_family_id, seed)
        if b < train:
            out[e.example_id] = SplitName.TRAIN
        elif b < train + validation:
            out[e.example_id] = SplitName.VALIDATION
        else:
            out[e.example_id] = SplitName.TEST
    return out


def transform_split_offenders(examples: list[Example], split_of: dict[str, SplitName]) -> list[str]:
    """example_ids whose split differs from their transformation parent's split.

    M1 has no augmentation (parents are all None), so this is empty; enforcing it now means a
    later transformation can never place a child in a different split from its parent.
    """
    known = {e.example_id for e in examples}
    offenders: list[str] = []
    for e in examples:
        parent = e.transformation_parent_id
        if parent and parent in known and split_of.get(e.example_id) != split_of.get(parent):
            offenders.append(e.example_id)
    return sorted(offenders)


__all__ = [
    "assign_family_splits",
    "assign_splits",
    "pair_sha",
    "transform_split_offenders",
    "union_find_groups",
]
