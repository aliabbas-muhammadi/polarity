"""Leakage audit — the gate that protects the sealed final holdout.

A sentence can appear in both train and an evaluation set through a different surface form
(re-cased, re-punctuated, reversed, or as a variant sharing a source family). Any of these
invalidates a held-out measurement. The audit reports every such overlap, and
``assert_holdout_isolated`` fails CLOSED: if final-holdout text reaches a train/val pair, a
training run must abort rather than silently report a contaminated number.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from polarity.data.schema import Example
from polarity.textnorm import normalize
from polarity.types import Pair, SplitManifestEntry


class LeakageError(RuntimeError):
    """Raised when the sealed final holdout is not isolated from training data."""


class SealedLeakageError(RuntimeError):
    """Raised when candidate data collides with a sealed/protected evaluation set."""


def _texts(p: Pair) -> tuple[str, str]:
    return (p.text_a, p.text_b)


class LeakageReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n_train: int
    n_holdout: int
    exact_overlaps: list[str]  # raw texts shared by both sides
    normalized_overlaps: list[str]  # normalized texts shared (beyond exact)
    reverse_pair_overlaps: list[str]  # holdout pair ids whose (b, a) is a train pair
    cross_split_groups: list[str]  # groups assigned to more than one split
    holdout_leaks: list[str]  # holdout pair ids with any text present in train

    @property
    def ok(self) -> bool:
        return not (
            self.exact_overlaps
            or self.normalized_overlaps
            or self.reverse_pair_overlaps
            or self.cross_split_groups
            or self.holdout_leaks
        )


def audit(
    train: Sequence[Pair],
    holdout: Sequence[Pair],
    entries: Sequence[SplitManifestEntry] | None = None,
) -> LeakageReport:
    """Compare a train set against a holdout set (and optionally a split manifest)."""
    train_raw = {t for p in train for t in _texts(p)}
    hold_raw = {t for p in holdout for t in _texts(p)}
    exact = sorted(train_raw & hold_raw)

    train_norm = {normalize(t) for t in train_raw}
    hold_norm = {normalize(t) for t in hold_raw}
    exact_norm = {normalize(t) for t in exact}
    normalized = sorted((train_norm & hold_norm) - exact_norm)

    train_pairs = {(p.text_a, p.text_b) for p in train}
    reverse = sorted(p.id for p in holdout if (p.text_b, p.text_a) in train_pairs)

    cross_split: list[str] = []
    if entries is not None:
        by_group: dict[str, set[str]] = defaultdict(set)
        for e in entries:
            by_group[e.group].add(e.split.value)
        cross_split = sorted(g for g, splits in by_group.items() if len(splits) > 1)

    leaks = sorted(p.id for p in holdout if any(normalize(t) in train_norm for t in _texts(p)))

    return LeakageReport(
        n_train=len(train),
        n_holdout=len(holdout),
        exact_overlaps=exact,
        normalized_overlaps=normalized,
        reverse_pair_overlaps=reverse,
        cross_split_groups=cross_split,
        holdout_leaks=leaks,
    )


def assert_holdout_isolated(train: Sequence[Pair], holdout: Sequence[Pair]) -> None:
    """Fail closed if any holdout text is present in the training data."""
    report = audit(train, holdout)
    if report.exact_overlaps or report.normalized_overlaps or report.holdout_leaks:
        offenders = report.holdout_leaks or report.exact_overlaps
        raise LeakageError(f"final holdout leaks into training data: {offenders[:5]}")


class SealedAuditReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n_candidates: int
    n_protected_texts: int
    text_collisions: list[str]  # candidate example_ids whose text hits a protected text
    reverse_pair_collisions: list[str]

    @property
    def ok(self) -> bool:
        return not (self.text_collisions or self.reverse_pair_collisions)


def sealed_set_audit(
    candidates: Sequence[Example], protected_pairs: Sequence[tuple[str, str]]
) -> SealedAuditReport:
    """Check training candidates against a sealed/protected set (e.g. the Veritas 75 pairs).

    Because source families are built from shared normalized text, any transitive family
    overlap implies at least one shared normalized text — so normalized-text collision plus
    reverse-pair collision together cover the family-overlap case as well.
    """
    protected_texts: set[str] = set()
    protected_pairset: set[tuple[str, str]] = set()
    for a, b in protected_pairs:
        na, nb = normalize(a), normalize(b)
        protected_texts.add(na)
        protected_texts.add(nb)
        protected_pairset.add((na, nb))

    text_hits: list[str] = []
    reverse_hits: list[str] = []
    for e in candidates:
        na, nb = normalize(e.text_a), normalize(e.text_b)
        if na in protected_texts or nb in protected_texts:
            text_hits.append(e.example_id)
        if (nb, na) in protected_pairset:
            reverse_hits.append(e.example_id)

    return SealedAuditReport(
        n_candidates=len(candidates),
        n_protected_texts=len(protected_texts),
        text_collisions=sorted(text_hits),
        reverse_pair_collisions=sorted(reverse_hits),
    )


def assert_sealed_isolated(
    candidates: Sequence[Example], protected_pairs: Sequence[tuple[str, str]]
) -> None:
    """Fail closed if any candidate collides with the protected set."""
    report = sealed_set_audit(candidates, protected_pairs)
    if not report.ok:
        offenders = report.text_collisions or report.reverse_pair_collisions
        raise SealedLeakageError(f"candidates collide with sealed set: {offenders[:5]}")


__all__ = [
    "LeakageError",
    "LeakageReport",
    "SealedAuditReport",
    "SealedLeakageError",
    "assert_holdout_isolated",
    "assert_sealed_isolated",
    "audit",
    "normalize",
    "sealed_set_audit",
]
