"""Shared text normalization for surface-form comparison and content hashing.

One definition used everywhere (leakage audit, dedup, source-family grouping, per-example
hashes) so "same text" means the same thing across the whole pipeline.
"""

from __future__ import annotations

import hashlib
import re

_PUNCT = re.compile(r"[^a-z0-9\s]")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Casefold, strip punctuation, and collapse whitespace."""
    return _WS.sub(" ", _PUNCT.sub("", text.lower())).strip()


def normalized_sha(text: str) -> str:
    """Hex sha256 of the normalized text — a stable, text-free identity for a string."""
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


__all__ = ["normalize", "normalized_sha"]
