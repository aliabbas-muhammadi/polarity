from pathlib import Path

import pytest

from polarity.train import selection as sel


def _rec(step: int, recall: float, loss: float = 0.5, ok: bool = True) -> sel.CheckpointRecord:
    return sel.CheckpointRecord(
        step=step,
        epoch=0,
        val_loss=loss,
        val_tau=0.9,
        val_recall_at_budget=recall,
        val_fpr_at_budget=0.008,
        val_pauc=0.4,
        satisfies_budget=ok,
    )


def test_select_prefers_recall_then_earliest_step():
    hist = [_rec(100, 0.10), _rec(200, 0.20), _rec(300, 0.20)]
    assert sel.select_checkpoint(hist).step == 200


def test_select_prefers_budget_satisfying_when_available():
    hist = [_rec(100, 0.99, ok=False), _rec(200, 0.10, ok=True)]
    assert (
        sel.select_checkpoint(hist).step == 200
    )  # the budget-satisfying one wins despite lower recall


def test_select_falls_back_when_none_satisfy():
    hist = [_rec(100, 0.10, ok=False), _rec(200, 0.30, ok=False)]
    assert sel.select_checkpoint(hist).step == 200


def test_incomplete_run_refuses_continuation_and_requires_restart(tmp_path: Path):
    (tmp_path / "best").mkdir()
    with pytest.raises(sel.RunDirError, match=r"restart.*fresh or cleared"):
        sel.assert_writable(tmp_path)


def test_run_dir_immutability(tmp_path: Path):
    sel.assert_writable(tmp_path)  # empty -> ok
    sel.mark_complete(tmp_path)
    assert sel.is_complete(tmp_path)
    with pytest.raises(sel.RunDirError, match="immutable"):
        sel.assert_writable(tmp_path)


def test_deterministic_subset_is_reproducible():
    ids = [f"id:{i}" for i in range(200)]
    a = sel.deterministic_subset(ids, 16, seed=42)
    b = sel.deterministic_subset(ids, 16, seed=42)
    assert a == b and len(a) == 16
    assert sel.deterministic_subset(ids, 16, seed=7) != a  # seed changes the subset
