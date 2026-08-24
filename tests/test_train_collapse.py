"""Collapse-detection tests (brief §4/§15). Pure and offline — no torch, no network.

The M3.2 mechanics gate: a constant-output model (M3/M3.1) must be flagged collapsed, a healthy
separable model must not, and non-finite scores are an instant collapse. Also proves the runtime
telemetry (CheckpointRecord score fields) round-trips and reconstructs a ScoreStats.
"""

from polarity.train import collapse as cm
from polarity.train.config import CollapseCfg
from polarity.train.selection import CheckpointRecord
from polarity.types import Category, Label, Prediction

CFG = CollapseCfg()


def _p(pid: str, gold: str, score: float) -> Prediction:
    return Prediction(
        pair_id=pid, gold=Label(gold), score=score, category=Category.PARAPHRASE, source="x"
    )


def _labelled(scores_by_class: list[tuple[str, float]]) -> list[Prediction]:
    return [_p(f"i{i}", g, s) for i, (g, s) in enumerate(scores_by_class)]


def test_constant_output_is_collapsed():
    # M3.1's exact failure: one score for everyone.
    preds = _labelled(
        [("CACHE_EQUIVALENT" if i % 2 else "NOT_CACHE_EQUIVALENT", 0.488) for i in range(200)]
    )
    st = cm.stats_from_predictions(preds)
    assert st.n_unique == 1 and st.std == 0.0 and st.class_separation == 0.0
    v = cm.assess(st, CFG)
    assert v.collapsed and v.reasons


def test_near_constant_output_is_collapsed():
    # M3's failure: two near-identical values, no class separation.
    preds = _labelled([("CACHE_EQUIVALENT", 0.472001), ("NOT_CACHE_EQUIVALENT", 0.472000)] * 100)
    assert cm.assess(cm.stats_from_predictions(preds), CFG).collapsed


def test_healthy_separable_model_is_not_collapsed():
    preds = _labelled(
        [
            ("CACHE_EQUIVALENT", 0.98) if i % 2 else ("NOT_CACHE_EQUIVALENT", 0.02)
            for i in range(200)
        ]
    )
    st = cm.stats_from_predictions(preds)
    assert st.n_unique >= 2 and st.class_separation is not None and st.class_separation > 0.9
    assert not cm.assess(st, CFG).collapsed


def test_nonfinite_scores_are_instant_collapse():
    st = cm.score_stats([float("nan"), 0.5, 0.5], [True, False, True])
    assert st.has_nonfinite
    assert cm.assess(st, CFG).collapsed
    assert not cm.is_finite(float("inf")) and not cm.is_finite(float("nan"))
    assert cm.is_finite(1.0)


def test_single_class_has_no_separation_signal_but_std_still_gates():
    # Only negatives present: class_separation is None, but a healthy spread is not collapsed.
    spread = cm.score_stats([i / 100 for i in range(100)], [False] * 100)
    assert spread.class_separation is None and not cm.assess(spread, CFG).collapsed
    flat = cm.score_stats([0.3] * 100, [False] * 100)
    assert cm.assess(flat, CFG).collapsed  # std + n_unique still catch a constant


def test_checkpoint_record_score_fields_roundtrip_and_reconstruct():
    st = cm.score_stats([0.9, 0.1, 0.8, 0.2], [True, False, True, False])
    rec = CheckpointRecord(
        step=100,
        epoch=0,
        val_loss=0.5,
        val_tau=0.7,
        val_recall_at_budget=0.4,
        val_fpr_at_budget=0.008,
        val_pauc=0.3,
        satisfies_budget=True,
        val_score_std=round(st.std, 6),
        val_n_unique=st.n_unique,
        val_score_mean_pos=round(st.mean_pos, 6),
        val_score_mean_neg=round(st.mean_neg, 6),
        val_class_separation=round(st.class_separation, 6),
        val_score_min=round(st.smin, 6),
        val_score_max=round(st.smax, 6),
        val_has_nonfinite=st.has_nonfinite,
        val_n_scored=st.n,
    )
    round_tripped = CheckpointRecord.model_validate_json(rec.model_dump_json())
    assert round_tripped == rec
    rebuilt = round_tripped.score_stats()
    assert rebuilt.n_unique == st.n_unique and rebuilt.n == st.n
    assert not cm.assess(rebuilt, CFG).collapsed  # this record is healthy


def test_old_record_without_score_fields_defaults_to_collapsed_stats():
    # Backward compat: a record serialized before the score fields existed must still parse.
    rec = CheckpointRecord(
        step=1,
        epoch=0,
        val_loss=0.5,
        val_tau=0.5,
        val_recall_at_budget=0.0,
        val_fpr_at_budget=0.0,
        val_pauc=0.0,
        satisfies_budget=False,
    )
    assert rec.val_n_unique == 0 and rec.val_score_std == 0.0
    assert cm.assess(rec.score_stats(), CFG).collapsed  # defaults read as collapsed (safe)
