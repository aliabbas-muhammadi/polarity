"""Deterministically (re)build the example split manifest over the synthetic fixtures.

Run: ``uv run python scripts/build_manifest.py``

Demonstrates the leakage-safe split primitive on committed synthetic data. The manifest is
immutable output: regenerate it only when the fixtures or the split algorithm change.
"""

from __future__ import annotations

import json
from pathlib import Path

from polarity.data.loaders import load_pairs
from polarity.data.split import assign_splits

FIXTURES = Path("data/fixtures/pairs.jsonl")
OUT = Path("manifests/synthetic-split.json")


def main() -> None:
    pairs = load_pairs(FIXTURES)
    entries = assign_splits(pairs, seed=42)
    payload = {
        "manifest_version": "polarity-split-1",
        "seed": 42,
        "ratios": {"train": 0.7, "validation": 0.15, "test": 0.15},
        "n": len(entries),
        "entries": [e.model_dump(mode="json") for e in entries],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(entries)} entries)")


if __name__ == "__main__":
    main()
