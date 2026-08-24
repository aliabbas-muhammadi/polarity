import json
from pathlib import Path

import pytest

from polarity.data.build import build
from polarity.eval.leakage import SealedLeakageError


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _golden(base: str = "veritas alpha question") -> str:
    return json.dumps(
        [
            {
                "id": "g1",
                "group": "gg",
                "base": base,
                "probe": "veritas beta question",
                "pairType": "paraphrase-should-hit",
                "equivalent": True,
                "note": "",
            }
        ]
    )


def _setup(tmp: Path) -> Path:
    raw = tmp / "raw"
    paws = "\n".join(
        json.dumps(
            {
                "id": i,
                "sentence1": f"paws sentence {i} alpha",
                "sentence2": f"paws sentence {i} beta",
                "label": i % 2,
            }
        )
        for i in range(1, 21)
    )
    _write(raw / "paws-wiki" / "train.jsonl", paws + "\n")

    can = "premise\thypothesis\tlabel\n" + "".join(
        f"cannot prem {i}\tcannot hyp {i}\t{i % 2}\n" for i in range(1, 21)
    )
    _write(raw / "cannot" / "cannot_dataset_v1.1.tsv", can)

    rows = [
        json.dumps(
            {
                "pairID": f"m{i}",
                "premise": f"mnli s1 {i}",
                "hypothesis": f"mnli s2 {i}",
                "label": 2 if i % 2 == 0 else 0,  # 2=contradiction (kept), 0=entailment (rejected)
            }
        )
        for i in range(1, 21)
    ]
    _write(raw / "mnli" / "train.jsonl", "\n".join(rows) + "\n")

    _write(
        raw / "nan-nli" / "nan.csv",
        "premise,hypothesis,label\n"
        "nan prem 1,nan hyp 1,contradiction\n"
        "nan prem 2,nan hyp 2,neutral\n",
    )
    _write(tmp / "data" / "snapshots" / "veritas-golden" / "golden.json", _golden())
    return raw


def _load(path: Path) -> list[dict[str, str]]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def test_build_end_to_end(tmp_path: Path):
    raw = _setup(tmp_path)
    summary = build(raw_root=raw, out_root=tmp_path, limit_per_source=100, seed=42)
    assert summary["status"] == "OK"
    assert summary["sealed_veritas_collisions"] == 0
    assert summary["remediation"] == "none_needed"

    for name in ("train", "validation", "internal_test"):
        assert (tmp_path / "manifests" / "primary" / f"{name}.jsonl").exists()
        assert (tmp_path / "manifests" / "with-mnli" / f"{name}.jsonl").exists()
    assert (tmp_path / "manifests" / "eval" / "nan-nli.jsonl").exists()

    for report in (
        "mapping_audit",
        "dedup",
        "leakage",
        "stats-primary",
        "stats-with-mnli",
        "recipe-comparison",
        "build-summary",
        "sampling",
    ):
        assert (tmp_path / "reports" / "m1" / f"{report}.json").exists()


def test_build_recipe_separability(tmp_path: Path):
    raw = _setup(tmp_path)
    build(raw_root=raw, out_root=tmp_path, limit_per_source=100, seed=42)

    def splits(recipe: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for name in ("train", "validation", "internal_test"):
            for e in _load(tmp_path / "manifests" / recipe / f"{name}.jsonl"):
                out[e["example_id"]] = e["split"]
        return out

    primary, with_mnli = splits("primary"), splits("with-mnli")
    assert set(primary) <= set(with_mnli)
    assert all(with_mnli[i] == primary[i] for i in primary)
    assert len(with_mnli) > len(primary)  # MNLI negatives are additive


def test_build_is_deterministic(tmp_path: Path):
    raw = _setup(tmp_path)
    build(raw_root=raw, out_root=tmp_path, limit_per_source=100, seed=42)
    first = (tmp_path / "manifests" / "with-mnli" / "train.jsonl").read_text()
    build(raw_root=raw, out_root=tmp_path, limit_per_source=100, seed=42)
    second = (tmp_path / "manifests" / "with-mnli" / "train.jsonl").read_text()
    assert first == second


def test_build_preserves_authoritative_release_attributions(tmp_path: Path):
    raw = _setup(tmp_path)
    attributions = tmp_path / "ATTRIBUTIONS.md"
    expected = "# Release attributions\n\nAuthoritative artifact-scope policy.\n"
    _write(attributions, expected)

    build(raw_root=raw, out_root=tmp_path, limit_per_source=100, seed=42)

    assert attributions.read_text(encoding="utf-8") == expected


def test_build_fails_closed_on_sealed_collision(tmp_path: Path):
    raw = _setup(tmp_path)
    # golden text now equals a PAWS training sentence -> must fail closed
    _write(
        tmp_path / "data" / "snapshots" / "veritas-golden" / "golden.json",
        _golden(base="paws sentence 1 alpha"),
    )
    with pytest.raises(SealedLeakageError):
        build(raw_root=raw, out_root=tmp_path, limit_per_source=100, seed=42)


def test_build_remediation_excludes_colliding_families(tmp_path: Path):
    raw = _setup(tmp_path)
    _write(
        tmp_path / "data" / "snapshots" / "veritas-golden" / "golden.json",
        _golden(base="paws sentence 1 alpha"),  # collides with a PAWS training sentence
    )
    summary = build(
        raw_root=raw,
        out_root=tmp_path,
        limit_per_source=100,
        seed=42,
        exclude_sealed_collisions=True,
    )
    assert summary["status"] == "OK"
    assert summary["excluded_families"] >= 1
    assert summary["remediation"] == "excluded_colliding_families"
    ids = {
        e["example_id"]
        for name in ("train", "validation", "internal_test")
        for e in _load(tmp_path / "manifests" / "with-mnli" / f"{name}.jsonl")
    }
    assert "paws-wiki:train:1" not in ids
