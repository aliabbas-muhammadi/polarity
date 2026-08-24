"""Author-run baseline inference -> committed predictions.

The ONE step that downloads models and uses CPU (needs the ``models`` optional-extra + local
``data/raw/``). Writes ``predictions/m2/<baseline>/{validation,internal_test}.jsonl`` and records
model provenance to ``reports/m2/models.json``. Everything downstream is keyless (see report.py).
"""

from __future__ import annotations

import json
from pathlib import Path

from polarity.baselines import deterministic
from polarity.baselines.materialize import RAW_ROOT, MaterializedPair, materialize
from polarity.baselines.registry import get_model
from polarity.baselines.schema import BaselinePrediction, shortcut_features

PRED_ROOT = Path("predictions/m2")
REPORTS_ROOT = Path("reports/m2")
PRIMARY = {
    "validation": Path("manifests/primary/validation.jsonl"),
    "internal_test": Path("manifests/primary/internal_test.jsonl"),
}
BASELINES = ("exact", "normalized", "embedding-cosine", "adaptive", "nli-cross-encoder")
_MODELS_FOR = {
    "embedding-cosine": ("bge-small",),
    "adaptive": ("bge-small",),
    "nli-cross-encoder": ("nli-deberta-v3-base",),
}


def _to_pred(mp: MaterializedPair, score: float) -> BaselinePrediction:
    cue, overlap, len_diff = shortcut_features(mp.text_a, mp.text_b)
    return BaselinePrediction(
        pair_id=mp.example_id,
        gold=mp.gold,
        score=round(float(score), 6),
        category=mp.category,
        source=mp.source,
        has_neg_cue=cue,
        token_overlap=overlap,
        len_diff=len_diff,
    )


def _write(path: Path, preds: list[BaselinePrediction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [p.model_dump_json() for p in sorted(preds, key=lambda x: x.pair_id)]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _score_stateless(
    name: str, val: list[MaterializedPair], test: list[MaterializedPair]
) -> tuple[list[float], list[float]]:
    if name == "exact":
        fn = deterministic.score_exact
    elif name == "normalized":
        fn = deterministic.score_normalized
    elif name == "embedding-cosine":
        from polarity.baselines.embedding import EmbeddingScorer

        emb = EmbeddingScorer(get_model("bge-small"))
        return (
            emb.score([(p.text_a, p.text_b) for p in val]),
            emb.score([(p.text_a, p.text_b) for p in test]),
        )
    elif name == "nli-cross-encoder":
        from polarity.baselines.cross_encoder import NliScorer

        nli = NliScorer(get_model("nli-deberta-v3-base"))
        return (
            nli.score([(p.text_a, p.text_b) for p in val]),
            nli.score([(p.text_a, p.text_b) for p in test]),
        )
    else:
        raise ValueError(f"unknown baseline {name!r}")
    return [fn(p.text_a, p.text_b) for p in val], [fn(p.text_a, p.text_b) for p in test]


def _record_models(name: str, out_root: Path) -> None:
    path = out_root / REPORTS_ROOT / "models.json"
    existing: dict[str, object] = json.loads(path.read_text()) if path.exists() else {}
    for key in _MODELS_FOR.get(name, ()):
        existing[key] = {**get_model(key).model_dump(mode="json"), "device": "cpu"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def run(name: str, *, out_root: Path = Path("."), raw_root: Path = RAW_ROOT) -> None:
    if name not in BASELINES:
        raise ValueError(f"unknown baseline {name!r} (known: {BASELINES})")
    val = materialize(out_root / PRIMARY["validation"], raw_root=raw_root)
    test = materialize(out_root / PRIMARY["internal_test"], raw_root=raw_root)

    if name == "adaptive":
        from polarity.baselines.adaptive import AdaptiveScorer

        scorer = AdaptiveScorer(get_model("bge-small")).fit(val)
        val_scores = scorer.score(val, loo=True)
        test_scores = scorer.score(test, loo=False)
    else:
        val_scores, test_scores = _score_stateless(name, val, test)

    _write(
        out_root / PRED_ROOT / name / "validation.jsonl",
        [_to_pred(p, s) for p, s in zip(val, val_scores, strict=True)],
    )
    _write(
        out_root / PRED_ROOT / name / "internal_test.jsonl",
        [_to_pred(p, s) for p, s in zip(test, test_scores, strict=True)],
    )
    _record_models(name, out_root)


__all__ = ["BASELINES", "PRED_ROOT", "run"]
