"""Deterministic exact / normalized baselines. Score in {0.0, 1.0}.

``normalize`` (textnorm) casefolds, strips punctuation, and collapses whitespace only — it KEEPS
negation tokens, numbers, entities, modality, and quantifiers, so it can never call a
meaning-changing pair equivalent by construction (brief §3).
"""

from __future__ import annotations

from polarity.textnorm import normalize


def score_exact(text_a: str, text_b: str) -> float:
    return 1.0 if text_a == text_b else 0.0


def score_normalized(text_a: str, text_b: str) -> float:
    return 1.0 if normalize(text_a) == normalize(text_b) else 0.0


__all__ = ["score_exact", "score_normalized"]
