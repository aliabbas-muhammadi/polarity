"""Content-addressed hashing for deterministic provenance and drift detection.

Recipe: JSON-able payload -> canonical JSON (sorted keys, no whitespace, ASCII, NaN/Inf
refused) -> sha256, prefixed with a scheme tag so the scheme can evolve without colliding
with older hashes. This is what makes split manifests, dataset snapshots, and the metrics
report byte-stable across machines and Python versions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SCHEME = "pl1"


def canonical_bytes(payload: Any) -> bytes:
    """Serialise a JSON-able payload to canonical, deterministic bytes."""
    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return text.encode("utf-8")


def content_hash(payload: Any) -> str:
    """Return the versioned, content-addressed hash of a JSON-able payload."""
    return f"{SCHEME}:{hashlib.sha256(canonical_bytes(payload)).hexdigest()}"


def sha256_file(path: Path) -> str:
    """Return the hex sha256 of a file's raw bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["SCHEME", "canonical_bytes", "content_hash", "sha256_file"]
