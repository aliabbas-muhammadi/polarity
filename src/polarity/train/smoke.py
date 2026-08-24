"""Local CPU smoke: exercise the whole training pipeline on a tiny deterministic subset (brief §22).

This proves the mechanics — tokenizer, batching, labels, loss, optimizer, checkpointing,
evaluation, prediction writing, calibration — before any paid GPU minute. It is emphatically **NOT
a Polarity result**: a few steps on ~64 pairs says nothing about model quality. Runs the real
deberta-v3-base on CPU (fp32) so it tests the actual tokenizer + head, and writes evidence to the
(gitignored) completed run dir. Interrupted training is restart-only; this smoke does not test
exact resume.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from polarity.baselines.materialize import RAW_ROOT
from polarity.train import model as model_mod
from polarity.train import runner
from polarity.train.config import TrainConfig, load_config
from polarity.train.predict import write_predictions
from polarity.train.selection import deterministic_subset, is_complete


def _tiny_manifest(src: Path, dst: Path, n: int, *, seed: int) -> None:
    rows = [json.loads(x) for x in src.read_text(encoding="utf-8").splitlines() if x.strip()]
    keep = set(deterministic_subset([r["example_id"] for r in rows], n, seed=seed))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(
        "\n".join(json.dumps(r) for r in rows if r["example_id"] in keep) + "\n", encoding="utf-8"
    )


def _smoke_cfg(cfg: TrainConfig) -> TrainConfig:
    tiny_training = cfg.training.model_copy(
        update={
            "max_length": 64,
            "batch_size": 8,
            "gradient_accumulation": 1,
            "epochs": 1,
            "eval_every_steps": 1,
            "early_stopping_patience": 99,
            "precision": "fp32",
        }
    )
    return cfg.model_copy(update={"training": tiny_training})


def run_smoke(
    cfg_path: Path = Path("configs/train/m3-deberta-v3-base.yaml"),
    run_dir: Path = Path("runs/smoke-m3"),
    *,
    raw_root: Path = RAW_ROOT,
) -> dict[str, Any]:
    cfg = _smoke_cfg(load_config(cfg_path))
    with tempfile.TemporaryDirectory(prefix="polarity-smoke-inputs-") as input_dir:
        inputs = Path(input_dir)
        train_m = inputs / "train.jsonl"
        val_m = inputs / "validation.jsonl"
        _tiny_manifest(Path("manifests/primary/train.jsonl"), train_m, 64, seed=cfg.training.seed)
        _tiny_manifest(
            Path("manifests/primary/validation.jsonl"), val_m, 32, seed=cfg.training.seed
        )

        s1 = runner.train(
            cfg,
            train_manifest=train_m,
            val_manifest=val_m,
            run_dir=run_dir,
            device="cpu",
            raw_root=raw_root,
            max_steps=3,
        )
        assert is_complete(run_dir), "single-session smoke run should complete"

        # Predict + write in the committed format (to the gitignored run dir, not predictions/m3).
        from polarity.baselines.materialize import materialize

        bundle = model_mod.load_checkpoint(run_dir / "best", cfg, device="cpu")
        val_pairs = materialize(val_m, raw_root=raw_root)
        scores = runner.predict_scores(bundle, val_pairs, cfg)
        write_predictions(val_pairs, scores, run_dir / "predictions" / "validation.jsonl")

    evidence = {
        "NOTICE": "SMOKE TEST — NOT A POLARITY RESULT (a few steps on ~64 pairs)",
        "training": s1,
        "interrupted_training": "restart-only; exact resume is unsupported",
        "predictions_written": len(scores),
        "params": s1.get("params"),
    }
    (run_dir / "EVIDENCE.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return evidence


__all__ = ["run_smoke"]


if __name__ == "__main__":
    print(json.dumps(run_smoke(), indent=2, sort_keys=True))
