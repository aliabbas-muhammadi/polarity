"""Reconstruct pair text for a manifest from the raw sources, verified against committed hashes.

Fails closed on any hash mismatch (raw drifted vs the committed manifest), and HARD-REFUSES any
path under the sealed Veritas snapshot — baseline code must never be able to consume the held-out
transfer set (docs/M2-METHODOLOGY.md; brief §18/§21).
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from polarity.data.adapters import adapt
from polarity.data.sources import get_source
from polarity.textnorm import normalized_sha
from polarity.types import Category, Label

_FORBIDDEN = "veritas-golden"
RAW_ROOT = Path("data/raw")


class MaterializeError(RuntimeError):
    """Raised on a refused path, a missing example, or a hash mismatch."""


class MaterializedPair(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    example_id: str
    text_a: str
    text_b: str
    gold: Label
    category: Category
    source: str


def _assert_not_veritas(path: Path | str) -> None:
    if _FORBIDDEN in str(path):
        raise MaterializeError(f"refusing to load the sealed Veritas set: {path}")


def _text_index(sources: set[str], raw_root: Path) -> dict[str, tuple[str, str]]:
    index: dict[str, tuple[str, str]] = {}
    for sid in sorted(sources):
        _assert_not_veritas(sid)
        examples, _ = adapt(get_source(sid), raw_root / sid)
        for e in examples:
            index[e.example_id] = (e.text_a, e.text_b)
    return index


def materialize(manifest_path: Path, *, raw_root: Path = RAW_ROOT) -> list[MaterializedPair]:
    """Reconstruct + verify the text for every row of a (non-Veritas) manifest."""
    _assert_not_veritas(manifest_path)
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sources = {str(r["source"]) for r in rows}
    for s in sources:
        _assert_not_veritas(s)

    index = _text_index(sources, raw_root)
    out: list[MaterializedPair] = []
    for r in rows:
        eid = r["example_id"]
        if eid not in index:
            raise MaterializeError(
                f"cannot reconstruct text for {eid} (raw missing? run `polarity data fetch`)"
            )
        text_a, text_b = index[eid]
        if normalized_sha(text_a) != r["a_sha"] or normalized_sha(text_b) != r["b_sha"]:
            raise MaterializeError(
                f"text hash mismatch for {eid} — raw drifted vs committed manifest"
            )
        if r["label"] is None:
            raise MaterializeError(f"{eid} has no label (an eval probe is not scorable here)")
        out.append(
            MaterializedPair(
                example_id=eid,
                text_a=text_a,
                text_b=text_b,
                gold=Label(r["label"]),
                category=Category(r["category"]),
                source=str(r["source"]),
            )
        )
    return out


__all__ = ["RAW_ROOT", "MaterializeError", "MaterializedPair", "materialize"]
