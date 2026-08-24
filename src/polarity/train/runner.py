"""The small, reliable custom training loop (brief §19/§20). Lazy-imports torch/transformers.

Deliberately not an orchestration framework: AdamW + linear-warmup schedule + optional bf16 autocast
+ grad clip, evaluated on VALIDATION every N updates against the frozen operational metric
(recall at FPR<=budget), checkpointing the best, and guarded by hard cost/runtime caps. Interrupted
runs are not resumed: restart in a fresh or cleared run directory. The pure selection/cost logic
lives in ``selection.py`` / ``cost.py`` and is unit-tested without torch; this module wires it to
real gradients and is exercised by the author-run CPU smoke (brief §22/§30).
"""

from __future__ import annotations

import csv
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from polarity.baselines import select
from polarity.baselines.materialize import RAW_ROOT, MaterializedPair, materialize
from polarity.eval import metrics
from polarity.eval.curves import partial_auc
from polarity.train import collapse as collapse_mod
from polarity.train import model as model_mod
from polarity.train.config import TrainConfig
from polarity.train.cost import CostGuard
from polarity.train.data import encode, labels_tensor
from polarity.train.predict import POSITIVE_INDEX
from polarity.train.selection import (
    CheckpointRecord,
    assert_writable,
    mark_complete,
    select_checkpoint,
)
from polarity.types import Prediction

BEST_DIR = "best"


def set_determinism(seed: int) -> None:
    import os

    import torch
    from transformers import set_seed

    os.environ.setdefault("PYTHONHASHSEED", str(seed))  # best-effort; env must be set before start
    set_seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def precision_for(config_precision: str, device: str, bf16_supported: bool) -> str:
    """Pure precision decision (torch-free, so the fp32/bf16 guarantee is unit-testable).

    bf16 autocast turns on ONLY when the config asks for it AND the hardware supports it on CUDA.
    An ``fp32`` config therefore trains in fp32 everywhere — the whole point of M3.2: bf16 cannot
    be silently re-enabled for an fp32 experiment (docs/M4-REPORT.md pins bf16 as the collapse
    suspect). CPU always falls back to fp32.
    """
    if device.startswith("cuda") and config_precision == "bf16" and bf16_supported:
        return "bf16"
    return "fp32"


def resolve_precision(cfg: TrainConfig, device: str) -> str:
    import torch

    supported = bool(device.startswith("cuda") and torch.cuda.is_bf16_supported())
    return precision_for(cfg.training.precision, device, supported)


def _minibatches(
    pairs: list[MaterializedPair], size: int, *, shuffle: bool, seed: int
) -> Iterator[list[MaterializedPair]]:
    order = list(range(len(pairs)))
    if shuffle:
        import random

        random.Random(seed).shuffle(order)
    for i in range(0, len(order), size):
        yield [pairs[j] for j in order[i : i + size]]


def predict_scores(
    bundle: model_mod.Bundle, pairs: list[MaterializedPair], cfg: TrainConfig
) -> list[float]:
    import torch

    bundle.model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for batch in _minibatches(pairs, max(cfg.training.batch_size, 1), shuffle=False, seed=0):
            enc = encode(bundle.tokenizer, batch, cfg).to(bundle.device)
            logits = bundle.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, POSITIVE_INDEX]
            scores.extend(float(p) for p in probs.tolist())
    return scores


def _as_predictions(pairs: list[MaterializedPair], scores: list[float]) -> list[Prediction]:
    return [
        Prediction(pair_id=p.example_id, gold=p.gold, score=s, category=p.category, source=p.source)
        for p, s in zip(pairs, scores, strict=True)
    ]


def evaluate(
    bundle: model_mod.Bundle,
    val: list[MaterializedPair],
    cfg: TrainConfig,
    *,
    step: int,
    epoch: int,
    val_loss: float,
) -> CheckpointRecord:
    preds = _as_predictions(val, predict_scores(bundle, val, cfg))
    budget = cfg.selection.fpr_budget
    tau = select.threshold_at_fpr(preds, budget)
    c = metrics.confusion_at(preds, tau)
    recall = c.tp / (c.tp + c.fn) if (c.tp + c.fn) else 0.0
    fpr = c.fp / (c.fp + c.tn) if (c.fp + c.tn) else 0.0
    st = collapse_mod.stats_from_predictions(preds)
    return CheckpointRecord(
        step=step,
        epoch=epoch,
        val_loss=round(val_loss, 6),
        val_tau=round(tau, 6),
        val_recall_at_budget=round(recall, 6),
        val_fpr_at_budget=round(fpr, 6),
        val_pauc=partial_auc(preds, cfg.selection.pauc_fpr_max),
        satisfies_budget=fpr <= budget + 1e-9,
        val_score_std=round(st.std, 6),
        val_n_unique=st.n_unique,
        val_score_mean_pos=round(st.mean_pos, 6) if st.mean_pos is not None else None,
        val_score_mean_neg=round(st.mean_neg, 6) if st.mean_neg is not None else None,
        val_class_separation=(
            round(st.class_separation, 6) if st.class_separation is not None else None
        ),
        val_score_min=round(st.smin, 6),
        val_score_max=round(st.smax, 6),
        val_has_nonfinite=st.has_nonfinite,
        val_n_scored=st.n,
    )


def _optimizer(model: Any, cfg: TrainConfig) -> Any:
    import torch

    decay: list[Any] = []
    no_decay: list[Any] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim == 1 or name.endswith(".bias")) else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.training.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=cfg.training.learning_rate,
        betas=(cfg.training.adam_beta1, cfg.training.adam_beta2),
        eps=cfg.training.adam_eps,
    )


def _log_row(run_dir: Path, row: dict[str, Any]) -> None:
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / "metrics.csv"
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def train(
    cfg: TrainConfig,
    *,
    train_manifest: Path,
    val_manifest: Path,
    run_dir: Path,
    device: str = "cpu",
    raw_root: Path = RAW_ROOT,
    max_steps: int | None = None,
    guard: CostGuard | None = None,
    interrupt_after_evals: int | None = None,
) -> dict[str, Any]:
    """Fine-tune, selecting the best checkpoint on validation. Returns a telemetry summary."""
    import torch

    assert_writable(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    set_determinism(cfg.training.seed)

    train_pairs = materialize(train_manifest, raw_root=raw_root)
    val_pairs = materialize(val_manifest, raw_root=raw_root)

    eff_batch = cfg.training.effective_batch
    updates_per_epoch = max(1, -(-len(train_pairs) // eff_batch))
    total_updates = max_steps if max_steps is not None else updates_per_epoch * cfg.training.epochs
    warmup = int(cfg.training.warmup_ratio * total_updates)

    precision = resolve_precision(cfg, device)
    bundle = model_mod.load_base(cfg, device=device)

    from transformers import get_scheduler

    optimizer = _optimizer(bundle.model, cfg)
    scheduler = get_scheduler(
        "linear", optimizer, num_warmup_steps=warmup, num_training_steps=total_updates
    )

    metric = cfg.selection.metric

    def _sel(rec: CheckpointRecord) -> float:
        # The selection scalar: pAUC (robust) or recall@FPR<=budget (can floor at 0 on a small
        # validation negative set — see docs/M3-REPORT.md).
        return rec.val_pauc if metric == "pauc" else rec.val_recall_at_budget

    history: list[CheckpointRecord] = []
    start_update = 0
    best_recall = -1.0  # best value of the selection metric

    started = time.monotonic()
    update = start_update
    bad_evals = 0
    evals_this_session = 0
    stop = False
    max_grad_norm = 0.0
    collapsed_streak = 0
    nonfinite_encountered = False
    stop_reason = ""

    for epoch in range(cfg.training.epochs):
        if stop or update >= total_updates:
            break
        micro = list(
            _minibatches(
                train_pairs, cfg.training.batch_size, shuffle=True, seed=cfg.training.seed + epoch
            )
        )
        bundle.model.train()
        accum: list[MaterializedPair] = []
        step_loss = 0.0
        n_micro = 0
        last_gnorm = 0.0
        for mb in micro:
            enc = encode(bundle.tokenizer, mb, cfg).to(bundle.device)
            labels = labels_tensor(mb).to(bundle.device)
            if precision == "bf16":
                with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
                    out = bundle.model(**enc, labels=labels)
            else:
                out = bundle.model(**enc, labels=labels)
            loss = out.loss / cfg.training.gradient_accumulation
            loss.backward()
            micro_loss = float(out.loss)
            step_loss += micro_loss
            n_micro += 1
            accum.append(mb[0])
            if not collapse_mod.is_finite(micro_loss):  # NaN/Inf loss = instant training failure
                nonfinite_encountered = True
                stop_reason = f"non-finite loss at update {update}"
                stop = True
                break
            if len(accum) >= cfg.training.gradient_accumulation:
                last_gnorm = float(
                    torch.nn.utils.clip_grad_norm_(
                        bundle.model.parameters(), cfg.training.grad_clip
                    )
                )
                max_grad_norm = max(max_grad_norm, last_gnorm)
                if not collapse_mod.is_finite(last_gnorm):  # NaN/Inf grad: do NOT step (corrupts)
                    nonfinite_encountered = True
                    stop_reason = f"non-finite gradient norm at update {update}"
                    stop = True
                    break
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                update += 1
                accum = []
                if update % cfg.training.eval_every_steps == 0 or update >= total_updates:
                    rec = evaluate(
                        bundle,
                        val_pairs,
                        cfg,
                        step=update,
                        epoch=epoch,
                        val_loss=step_loss / max(1, n_micro),  # mean training CE over the window
                    )
                    history.append(rec)
                    verdict = collapse_mod.assess(rec.score_stats(), cfg.collapse)
                    collapsed_streak = collapsed_streak + 1 if verdict.collapsed else 0
                    _log_row(
                        run_dir,
                        {
                            "update": update,
                            "epoch": epoch,
                            "train_loss": rec.val_loss,  # mean training CE (not a validation loss)
                            "grad_norm": round(last_gnorm, 4),
                            "val_recall": rec.val_recall_at_budget,
                            "val_fpr": rec.val_fpr_at_budget,
                            "val_tau": rec.val_tau,
                            "val_pauc": rec.val_pauc,
                            "val_score_std": rec.val_score_std,
                            "val_n_unique": rec.val_n_unique,
                            "val_class_sep": rec.val_class_separation,
                            "collapsed": verdict.collapsed,
                            "lr": round(scheduler.get_last_lr()[0], 8),
                            "wall_s": round(time.monotonic() - started, 2),
                        },
                    )
                    step_loss = 0.0
                    n_micro = 0
                    improved = _sel(rec) > best_recall + 1e-12
                    if improved:
                        best_recall = _sel(rec)
                        model_mod.save_checkpoint(bundle, run_dir / BEST_DIR)
                        bad_evals = 0
                    else:
                        bad_evals += 1
                    if guard is not None:
                        guard.check_elapsed((time.monotonic() - started) / 3600.0)
                    evals_this_session += 1
                    if (
                        interrupt_after_evals is not None
                        and evals_this_session >= interrupt_after_evals
                    ):
                        # Simulate a timeout/cost-cap interruption. The run remains incomplete and
                        # must be restarted in a fresh or cleared directory.
                        return {
                            "experiment_id": cfg.experiment_id,
                            "interrupted": True,
                            "updates_run": update,
                            "wall_seconds": round(time.monotonic() - started, 2),
                        }
                    # Hard collapse gate (brief §4): terminate a paid run whose validation output
                    # is *sustainedly* collapsed (constant / no class separation), but only after a
                    # warmup so a slow-starting real model is never killed on one noisy checkpoint.
                    if (
                        len(history) > cfg.collapse.warmup_evals
                        and collapsed_streak >= cfg.collapse.patience_evals
                    ):
                        stop_reason = (
                            f"sustained collapse: {collapsed_streak} consecutive collapsed "
                            f"checkpoints ({', '.join(verdict.reasons)})"
                        )
                        stop = True
                        break
                    if bad_evals >= cfg.training.early_stopping_patience or update >= total_updates:
                        stop = True
                        break

    if not history:  # ensure at least one eval + a best checkpoint exist
        rec = evaluate(bundle, val_pairs, cfg, step=update, epoch=0, val_loss=0.0)
        history.append(rec)
        model_mod.save_checkpoint(bundle, run_dir / BEST_DIR)
    best = select_checkpoint(history, metric=metric)
    best_verdict = collapse_mod.assess(best.score_stats(), cfg.collapse)
    mark_complete(run_dir)
    return {
        "experiment_id": cfg.experiment_id,
        "device": device,
        "precision_requested": cfg.training.precision,
        "precision_used": precision,
        "total_updates_planned": total_updates,
        "updates_run": update,
        "n_train": len(train_pairs),
        "n_val": len(val_pairs),
        "params": model_mod.count_params(bundle.model),
        "best_checkpoint": str(run_dir / BEST_DIR),
        "best": best.model_dump(mode="json"),
        "max_grad_norm": round(max_grad_norm, 4),
        "nonfinite_encountered": nonfinite_encountered,
        "stop_reason": stop_reason,
        "collapse": {
            "thresholds": cfg.collapse.model_dump(mode="json"),
            "selected_checkpoint": best_verdict.as_dict(),
            "final_collapsed_streak": collapsed_streak,
        },
        "wall_seconds": round(time.monotonic() - started, 2),
        "history": [h.model_dump(mode="json") for h in history],
    }


__all__ = [
    "evaluate",
    "precision_for",
    "predict_scores",
    "resolve_precision",
    "set_determinism",
    "train",
]
