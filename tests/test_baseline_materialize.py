import json
from pathlib import Path

import pytest

from polarity.baselines.materialize import MaterializeError, materialize
from polarity.textnorm import normalized_sha


def _raw(tmp: Path) -> Path:
    raw = tmp / "raw"
    (raw / "paws-wiki").mkdir(parents=True)
    rows = [
        {
            "id": 1,
            "sentence1": "Cat sat on the mat",
            "sentence2": "The cat sat on a mat",
            "label": 1,
        },
        {"id": 2, "sentence1": "Dogs bark loudly", "sentence2": "Cats meow softly", "label": 0},
    ]
    (raw / "paws-wiki" / "train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    return raw


def _row(eid: str, a: str, b: str, label: str, category: str = "paraphrase") -> dict[str, object]:
    return {
        "example_id": eid,
        "source": "paws-wiki",
        "family": "fam:x",
        "split": "validation",
        "label": label,
        "category": category,
        "tier": "primary",
        "a_sha": normalized_sha(a),
        "b_sha": normalized_sha(b),
    }


def test_reconstructs_and_verifies(tmp_path: Path):
    raw = _raw(tmp_path)
    manifest = tmp_path / "m.jsonl"
    rows = [
        _row("paws-wiki:train:1", "Cat sat on the mat", "The cat sat on a mat", "CACHE_EQUIVALENT"),
        _row(
            "paws-wiki:train:2",
            "Dogs bark loudly",
            "Cats meow softly",
            "NOT_CACHE_EQUIVALENT",
            "unknown",
        ),
    ]
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = materialize(manifest, raw_root=raw)
    assert len(out) == 2
    assert out[0].text_a == "Cat sat on the mat"


def test_hash_mismatch_fails_closed(tmp_path: Path):
    raw = _raw(tmp_path)
    manifest = tmp_path / "m.jsonl"
    row = _row(
        "paws-wiki:train:1", "Cat sat on the mat", "The cat sat on a mat", "CACHE_EQUIVALENT"
    )
    row["a_sha"] = "deadbeef"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(MaterializeError):
        materialize(manifest, raw_root=raw)


def test_refuses_veritas_manifest_path(tmp_path: Path):
    path = tmp_path / "data" / "snapshots" / "veritas-golden" / "golden.json"
    path.parent.mkdir(parents=True)
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(MaterializeError):
        materialize(path, raw_root=tmp_path)


def test_refuses_veritas_source(tmp_path: Path):
    manifest = tmp_path / "m.jsonl"
    row = {
        "example_id": "veritas-golden:x",
        "source": "veritas-golden",
        "family": "f",
        "split": "validation",
        "label": "CACHE_EQUIVALENT",
        "category": "paraphrase",
        "tier": "primary",
        "a_sha": "x",
        "b_sha": "y",
    }
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(MaterializeError):
        materialize(manifest, raw_root=tmp_path)
