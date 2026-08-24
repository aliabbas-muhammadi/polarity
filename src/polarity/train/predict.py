"""Frozen-checkpoint scoring -> committed M3 predictions, in the exact M2 prediction format.

The write path is pure (reused by the offline schema test): one ``BaselinePrediction`` per pair —
the M0 ``Prediction`` fields plus the shortcut features — sorted by ``pair_id``, scores rounded to
6dp, trailing newline. So the whole keyless report/drift-gate machinery reads M3 predictions with
zero new plumbing. Scoring itself lazy-imports torch (author-run, brief §12).
"""

from __future__ import annotations

from pathlib import Path

from polarity.baselines.materialize import MaterializedPair
from polarity.baselines.schema import BaselinePrediction, shortcut_features
from polarity.types import Label

PRED_ROOT = Path("predictions/m3")


def to_prediction(pair: MaterializedPair, score: float) -> BaselinePrediction:
    cue, overlap, len_diff = shortcut_features(pair.text_a, pair.text_b)
    return BaselinePrediction(
        pair_id=pair.example_id,
        gold=pair.gold,
        score=round(float(score), 6),
        category=pair.category,
        source=pair.source,
        has_neg_cue=cue,
        token_overlap=overlap,
        len_diff=len_diff,
    )


def write_predictions(pairs: list[MaterializedPair], scores: list[float], path: Path) -> None:
    preds = [to_prediction(p, s) for p, s in zip(pairs, scores, strict=True)]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [p.model_dump_json() for p in sorted(preds, key=lambda x: x.pair_id)]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


#: Fixed label convention: head index 1 = CACHE_EQUIVALENT (positive), 0 = NOT. The score written
#: to predictions is softmax[:, POSITIVE_INDEX] = P(CACHE_EQUIVALENT). Not derived from enum
#: ordering (that would be a silent footgun); the loss labels in train/data.py use the same map.
POSITIVE_INDEX = 1


def label_index(gold: Label) -> int:
    return POSITIVE_INDEX if gold == Label.CACHE_EQUIVALENT else 1 - POSITIVE_INDEX


def predict_split(
    cfg_path: Path,
    checkpoint_dir: Path,
    manifest: Path,
    out_path: Path,
    *,
    device: str = "cpu",
    raw_root: Path | None = None,
) -> int:
    """Author-run: load the frozen checkpoint, score one split, write predictions. Lazy torch.

    ``device`` must match where the run happened — on a GPU pod pass ``"cuda"``; CPU inference
    with a high host thread count thrashes and effectively hangs.
    """
    from polarity.baselines.materialize import RAW_ROOT, materialize
    from polarity.train import model as model_mod
    from polarity.train import runner
    from polarity.train.config import load_config

    cfg = load_config(cfg_path)
    pairs = materialize(manifest, raw_root=raw_root or RAW_ROOT)
    bundle = model_mod.load_checkpoint(checkpoint_dir, cfg, device=device)
    scores = runner.predict_scores(bundle, pairs, cfg)
    write_predictions(pairs, scores, out_path)
    return len(pairs)


__all__ = [
    "POSITIVE_INDEX",
    "PRED_ROOT",
    "label_index",
    "predict_split",
    "to_prediction",
    "write_predictions",
]
