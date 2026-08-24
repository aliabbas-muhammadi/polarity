"""Deterministic semantic categorizer for real data (the docs/SPEC.md taxonomy).

Assigns each pair a Category from cheap lexical signals only — no model, no threshold. Where
the evidence is insufficient it returns ``UNKNOWN`` rather than forcing a label: the point of
M1 is to MEASURE the coverage we actually have, not to manufacture it. The negation / antonym /
scope lexicons are ported from the veritas deterministic guard.

The categorizer uses the (already-assigned) cache-equivalence label as a prior:
- an equivalent pair carrying a negation cue or negative morphology is EQUIVALENT_NEGATION —
  the load-bearing "not safe == insecure" slice the whole project must not misread;
- a non-equivalent pair is bucketed by the signal that flips its meaning.
"""

from __future__ import annotations

import re

from polarity.types import Category, Label

_TOKEN = re.compile(r"[a-z0-9']+")
_NUM = re.compile(r"\d+(?:\.\d+)?")

NEGATION_CUES = frozenset(
    {
        "not",
        "no",
        "never",
        "none",
        "neither",
        "nor",
        "isn't",
        "aren't",
        "don't",
        "doesn't",
        "didn't",
        "won't",
        "shouldn't",
        "wouldn't",
        "couldn't",
        "wasn't",
        "weren't",
        "isnt",
        "arent",
        "dont",
        "doesnt",
        "didnt",
        "wont",
        "shouldnt",
        "wouldnt",
        "couldnt",
        "wasnt",
        "werent",
    }
)

OPPOSITES: tuple[tuple[frozenset[str], frozenset[str]], ...] = (
    (frozenset({"good"}), frozenset({"bad"})),
    (frozenset({"safe"}), frozenset({"unsafe", "dangerous"})),
    (frozenset({"legal"}), frozenset({"illegal"})),
    (frozenset({"healthy"}), frozenset({"unhealthy"})),
    (frozenset({"true"}), frozenset({"false"})),
    (frozenset({"up"}), frozenset({"down"})),
    (frozenset({"open"}), frozenset({"closed", "close"})),
    (
        frozenset({"raise", "raises", "increase", "increases", "higher", "rise"}),
        frozenset({"lower", "lowers", "decrease", "decreases", "reduce", "fall"}),
    ),
    (
        frozenset({"allowed", "permitted"}),
        frozenset({"prohibited", "banned", "forbidden"}),
    ),
    (frozenset({"effective"}), frozenset({"ineffective"})),
    (
        frozenset({"include", "includes", "including"}),
        frozenset({"exclude", "excludes", "excluding"}),
    ),
    (frozenset({"compiled"}), frozenset({"interpreted"})),
    (frozenset({"enable", "enabled"}), frozenset({"disable", "disabled"})),
    (frozenset({"valid"}), frozenset({"invalid"})),
)

SCOPE_CLASSES: tuple[frozenset[str], ...] = (
    frozenset({"today", "tomorrow", "yesterday", "tonight"}),
    frozenset(
        {
            "child",
            "children",
            "kid",
            "kids",
            "infant",
            "infants",
            "baby",
            "toddler",
            "adult",
            "adults",
            "teen",
            "teenager",
            "elderly",
            "senior",
            "seniors",
        }
    ),
    frozenset({"celsius", "fahrenheit", "kelvin"}),
    frozenset(
        {
            "km",
            "kilometer",
            "kilometers",
            "mile",
            "miles",
            "meter",
            "meters",
            "feet",
            "foot",
        }
    ),
    frozenset({"soccer", "basketball", "baseball", "football", "hockey", "tennis", "cricket"}),
)

QUANTIFIERS = frozenset({"all", "every", "some", "any", "none", "most", "few", "each", "no"})

_NEG_PREFIXES = ("un", "in", "im", "il", "ir", "non", "dis")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def _nums(text: str) -> set[str]:
    return set(_NUM.findall(text))


def _negation_parity_differs(ta: set[str], tb: set[str]) -> bool:
    return any((cue in ta) != (cue in tb) for cue in NEGATION_CUES)


def _antonym_flip(ta: set[str], tb: set[str]) -> bool:
    for side_a, side_b in OPPOSITES:
        if (ta & side_a and tb & side_b) or (ta & side_b and tb & side_a):
            return True
    return False


def _scope_flip(ta: set[str], tb: set[str]) -> bool:
    for cls in SCOPE_CLASSES:
        ma, mb = sorted(ta & cls), sorted(tb & cls)
        if ma and mb and ma != mb:
            return True
    return False


def _number_mismatch(a: str, b: str) -> bool:
    na, nb = _nums(a), _nums(b)
    return bool(na) and bool(nb) and na != nb


def _quantifier_change(ta: set[str], tb: set[str]) -> bool:
    qa, qb = ta & QUANTIFIERS, tb & QUANTIFIERS
    return bool(qa or qb) and qa != qb


def _morphological_negation(ta: set[str], tb: set[str]) -> bool:
    """One side has a prefix-negated form of a stem present on the other (insecure/secure)."""
    for src, dst in ((ta, tb), (tb, ta)):
        for tok in src:
            for pre in _NEG_PREFIXES:
                if tok.startswith(pre) and len(tok) > len(pre) + 2 and tok[len(pre) :] in dst:
                    return True
    return False


def categorize(text_a: str, text_b: str, label: Label | None) -> Category:
    """Best-effort deterministic category; UNKNOWN when evidence is insufficient."""
    ta, tb = _tokens(text_a), _tokens(text_b)

    if text_a.strip().lower() == text_b.strip().lower():
        return Category.EXACT_REPEAT

    if label == Label.CACHE_EQUIVALENT:
        if (NEGATION_CUES & ta) or (NEGATION_CUES & tb) or _morphological_negation(ta, tb):
            return Category.EQUIVALENT_NEGATION
        union = ta | tb
        overlap = len(ta & tb) / len(union) if union else 0.0
        return Category.HARD_PARAPHRASE if overlap < 0.4 else Category.PARAPHRASE

    # NOT_CACHE_EQUIVALENT or unlabelled (eval probe) — bucket by the flipping signal.
    if _negation_parity_differs(ta, tb):
        return Category.EXPLICIT_NEGATION
    if _antonym_flip(ta, tb):
        return Category.LEXICAL_CONTRADICTION
    if _number_mismatch(text_a, text_b):
        return Category.NUMERIC_CHANGE
    if _scope_flip(ta, tb):
        return Category.SCOPE_CHANGE
    if _quantifier_change(ta, tb):
        return Category.QUANTIFIER_CHANGE
    return Category.UNKNOWN


__all__ = ["categorize"]
