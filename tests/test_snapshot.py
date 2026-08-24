from pathlib import Path

from polarity.data.loaders import golden_to_pairs, load_golden, load_pairs
from polarity.data.registry import read_provenance
from polarity.eval.leakage import audit
from polarity.fingerprint import sha256_file
from polarity.types import Label

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "data" / "snapshots" / "veritas-golden"


def test_snapshot_row_count():
    assert len(load_golden(SNAP / "golden.json")) == 75


def test_checksum_matches_provenance():
    rec = read_provenance(SNAP / "PROVENANCE.json")
    assert rec.sha256 == sha256_file(SNAP / "golden.json")
    assert rec.rows == 75
    assert rec.sealed is True


def test_label_balance():
    pairs = golden_to_pairs(load_golden(SNAP / "golden.json"))
    equivalent = sum(1 for p in pairs if p.label == Label.CACHE_EQUIVALENT)
    assert equivalent == 31
    assert len(pairs) - equivalent == 44


def test_holdout_isolated_from_fixtures():
    train = load_pairs(ROOT / "data" / "fixtures" / "pairs.jsonl")
    holdout = golden_to_pairs(load_golden(SNAP / "golden.json"))
    assert audit(train, holdout).ok
