"""Pure checkpoint-selection and run-directory logic — no torch, fully offline-testable.

The torch training loop (runner.py) calls these; keeping them pure lets the test suite verify the
frozen selection rule, run-dir immutability, and the deterministic smoke
subset without importing torch or downloading a model — the same seam M2 used for its scorers.

Checkpoint selection (frozen before training, brief §9): among validation snapshots, pick the one
with the highest recall at the FPR≤budget threshold; break ties by the earliest step, then the
lowest validation loss. Selection reads VALIDATION only — internal_test never enters here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from polarity.train.collapse import ScoreStats

DONE_MARKER = "DONE"


class RunDirError(RuntimeError):
    """Raised when a run directory would be overwritten (immutability, brief §19)."""


class CheckpointRecord(BaseModel):
    """A validation-only snapshot for one evaluated step."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int
    epoch: int
    val_loss: float
    val_tau: float  # validation threshold at FPR<=budget
    val_recall_at_budget: float  # validation recall at that threshold
    val_fpr_at_budget: float
    val_pauc: float  # partial AUC of the FP-recall curve over FPR in [0, pauc_max] (stability)
    satisfies_budget: bool  # achieved validation FPR <= budget
    # Validation score-distribution telemetry (brief §4, defaulted for backward compatibility).
    # Populated by runner.evaluate(); drives the runtime collapse gate + the M3.1-vs-M3.2 dynamics.
    val_score_std: float = 0.0
    val_n_unique: int = 0
    val_score_mean_pos: float | None = None
    val_score_mean_neg: float | None = None
    val_class_separation: float | None = None
    val_score_min: float = 0.0
    val_score_max: float = 0.0
    val_has_nonfinite: bool = False
    val_n_scored: int = 0

    def score_stats(self) -> ScoreStats:
        return ScoreStats(
            n=self.val_n_scored,
            std=self.val_score_std,
            n_unique=self.val_n_unique,
            mean_pos=self.val_score_mean_pos,
            mean_neg=self.val_score_mean_neg,
            class_separation=self.val_class_separation,
            smin=self.val_score_min,
            smax=self.val_score_max,
            has_nonfinite=self.val_has_nonfinite,
        )


def select_checkpoint(
    history: list[CheckpointRecord], *, metric: str = "recall_at_fpr"
) -> CheckpointRecord:
    """Frozen lexicographic rule keyed by the selection metric.

    ``recall_at_fpr`` (default): prefer budget-satisfying steps, then max recall@FPR<=budget — but
    this floors at 0 on a small validation negative set (see docs/M3-REPORT.md). ``pauc``: max
    validation partial-AUC over FPR in [0, pauc_max] (robust to the thin operating point). Ties in
    both break by earliest step, then lowest loss.
    """
    if not history:
        raise ValueError("empty checkpoint history")
    if metric == "pauc":
        return min(history, key=lambda h: (-h.val_pauc, h.step, h.val_loss))
    pool = [h for h in history if h.satisfies_budget] or list(history)
    return min(pool, key=lambda h: (-h.val_recall_at_budget, h.step, h.val_loss))


def is_complete(run_dir: Path) -> bool:
    return (run_dir / DONE_MARKER).exists()


def assert_writable(run_dir: Path) -> None:
    """Require a fresh or cleared run directory; no exact-resume capability is provided."""
    if is_complete(run_dir):
        raise RunDirError(
            f"run dir {run_dir} is complete (has {DONE_MARKER}); runs are immutable — use a new id"
        )
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RunDirError(
            f"run dir {run_dir} contains an incomplete run; exact resume is unsupported — "
            "restart in a fresh or cleared run directory"
        )


def mark_complete(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / DONE_MARKER).write_text("", encoding="utf-8")


def _rank_key(seed: int, example_id: str) -> str:
    return hashlib.sha256(f"{seed}:{example_id}".encode()).hexdigest()


def deterministic_subset(example_ids: list[str], n: int, *, seed: int) -> list[str]:
    """A reproducible size-n subset (stable hash order) for the smoke run. No RNG state."""
    ordered = sorted(example_ids, key=lambda eid: _rank_key(seed, eid))
    return ordered[: min(n, len(ordered))]


def demo() -> None:
    hist = [
        CheckpointRecord(
            step=s,
            epoch=0,
            val_loss=loss,
            val_tau=0.9,
            val_recall_at_budget=rec,
            val_fpr_at_budget=0.008,
            val_pauc=0.5,
            satisfies_budget=True,
        )
        for s, rec, loss in [(100, 0.10, 0.5), (200, 0.20, 0.4), (300, 0.20, 0.45)]
    ]
    best = select_checkpoint(hist)
    assert best.step == 200, best.step  # highest recall, earliest on tie
    sub = deterministic_subset([f"id:{i}" for i in range(100)], 8, seed=42)
    assert len(sub) == 8 and sub == deterministic_subset(
        [f"id:{i}" for i in range(100)], 8, seed=42
    )
    print("train.selection demo OK")


__all__ = [
    "DONE_MARKER",
    "CheckpointRecord",
    "RunDirError",
    "assert_writable",
    "deterministic_subset",
    "is_complete",
    "mark_complete",
    "select_checkpoint",
]


if __name__ == "__main__":
    demo()
