"""Training-signal diagnostics (M5). The load-bearing test is overfit-one-batch: a correctly
wired classifier memorizes a tiny labelled set; a broken pipeline cannot.

M5 traced the M3/M3.1/M3.2 collapse to a **transformers 5.x regression in DebertaV2 fine-tuning**
(docs/M5-REPORT.md): under transformers 5.x the model converges to a constant, input-invariant
predictor; under 4.x (tested 4.46.3, 4.57.6) it learns normally, and distilbert learns through the
identical pipeline under both. So the version guard here is the durable protection — it refuses to
start a (paid) training run on a transformers major known to break DeBERTa fine-tuning.

The overfit itself lazy-imports torch/transformers and is author-run (needs the ``train`` extra);
the pure helpers (version guard, verdict, trivial pairs) are torch-free and unit-tested.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polarity.baselines.materialize import RAW_ROOT, MaterializedPair, materialize
from polarity.train.config import TrainConfig
from polarity.train.predict import POSITIVE_INDEX
from polarity.train.selection import deterministic_subset
from polarity.types import Category, Label

# transformers majors that train DeBERTa-v2 correctly (see module docstring). Tested good: 4.46.3
# and 4.57.6; known broken: 5.15.1. Floor keeps DeBERTa-v3 support; ceiling excludes the regression.
TRANSFORMERS_MIN = (4, 44)
TRANSFORMERS_MAX_EXCLUSIVE = (5, 0)


def _parse_version(v: str) -> tuple[int, int]:
    parts = v.split(".")
    return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0


def transformers_supported(version: str) -> bool:
    """True if ``version`` is in the DeBERTa-fine-tuning-safe range [4.44, 5.0)."""
    mm = _parse_version(version)
    return TRANSFORMERS_MIN <= mm < TRANSFORMERS_MAX_EXCLUSIVE


def assert_transformers_supported() -> str:
    """Refuse to proceed on a transformers version known to break DeBERTa-v2 fine-tuning (M5).

    Returns the version string on success. Called at the start of the paid training path so no
    GPU minute is ever spent on the collapse again.
    """
    import transformers

    v = str(transformers.__version__)
    if not transformers_supported(v):
        lo = f"{TRANSFORMERS_MIN[0]}.{TRANSFORMERS_MIN[1]}"
        hi = f"{TRANSFORMERS_MAX_EXCLUSIVE[0]}.{TRANSFORMERS_MAX_EXCLUSIVE[1]}"
        raise RuntimeError(
            f"transformers {v} is outside the DeBERTa-fine-tuning-safe range [{lo}, {hi}). "
            "transformers 5.x collapses DeBERTa-v2 fine-tuning to a constant predictor "
            "(docs/M5-REPORT.md). Install a supported version: `uv sync --extra train` "
            "resolves transformers <5 via the pinned `train` extra."
        )
    return v


def trivial_pairs(n_per_class: int = 8) -> list[MaterializedPair]:
    """A deliberately trivial, mechanically-separable batch: A/A = equivalent, A/Z = not.

    A correctly wired model must memorize this in a few dozen steps. If it cannot, the bug is in the
    pipeline/model/objective — not the (real) data.
    """
    out: list[MaterializedPair] = []
    for i in range(n_per_class):
        out.append(
            MaterializedPair(
                example_id=f"triv-pos-{i}",
                text_a=f"alpha token number {i}",
                text_b=f"alpha token number {i}",
                gold=Label.CACHE_EQUIVALENT,
                category=Category.PARAPHRASE,
                source="diagnostic-synthetic",
            )
        )
        out.append(
            MaterializedPair(
                example_id=f"triv-neg-{i}",
                text_a=f"alpha token number {i}",
                text_b=f"zeta symbol distinct {i}",
                gold=Label.NOT_CACHE_EQUIVALENT,
                category=Category.EXPLICIT_NEGATION,
                source="diagnostic-synthetic",
            )
        )
    return out


def real_balanced_subset(
    manifest: Path, n_per_class: int = 8, *, raw_root: Path = RAW_ROOT
) -> list[MaterializedPair]:
    """A tiny balanced subset of REAL train pairs (deterministic; never internal_test)."""
    pairs = materialize(manifest, raw_root=raw_root)
    by_id = {p.example_id: p for p in pairs}
    pos = [p.example_id for p in pairs if p.gold == Label.CACHE_EQUIVALENT]
    neg = [p.example_id for p in pairs if p.gold == Label.NOT_CACHE_EQUIVALENT]
    keep = deterministic_subset(pos, n_per_class, seed=42) + deterministic_subset(
        neg, n_per_class, seed=42
    )
    return [by_id[i] for i in keep]


@dataclass
class OverfitResult:
    transformers_version: str
    device: str
    n_pairs: int
    steps: int
    lr: float
    prior_ce: float
    final_loss: float
    final_accuracy: float
    prob_min: float
    prob_max: float
    prob_spread: float
    pooled_repr_std: float
    max_grad_norm: float
    delta_head_norm: float
    delta_backbone_norm: float
    verdict: str
    history: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# A correctly wired model separates the classes and drives CE well below the class prior.
def _verdict(final_acc: float, final_loss: float, spread: float) -> str:
    return (
        "OVERFIT_PASS"
        if (final_acc >= 0.95 and final_loss < 0.35 and spread > 0.5)
        else "OVERFIT_FAIL"
    )


def _prior_ce(labels: Any) -> float:
    p = float(labels.float().mean())
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def overfit_one_batch(
    cfg: TrainConfig,
    pairs: list[MaterializedPair],
    *,
    steps: int = 150,
    lr: float = 1e-5,
    warmup: int = 15,
    device: str = "cpu",
    seed: int = 42,
) -> OverfitResult:
    """Repeatedly train the REAL model/encode/label path on one fixed tiny batch. No early stop, no
    collapse gate — the point is to overfit. Exercises model.load_base + data.encode + labels."""
    import torch
    from transformers import get_scheduler

    from polarity.train import model as model_mod
    from polarity.train.data import encode, labels_tensor

    tv = assert_transformers_supported()
    torch.manual_seed(seed)
    bundle = model_mod.load_base(cfg, device=device)
    enc = encode(bundle.tokenizer, pairs, cfg).to(device)
    labels = labels_tensor(pairs).to(device)
    prior = _prior_ce(labels)

    head = next(p for n, p in bundle.model.named_parameters() if "classifier" in n and p.ndim == 2)
    bb = next(
        p for n, p in bundle.model.named_parameters() if "encoder.layer.0." in n and p.ndim == 2
    )
    pre_head, pre_bb = head.detach().clone(), bb.detach().clone()

    opt = torch.optim.AdamW(
        bundle.model.parameters(),
        lr=lr,
        betas=(cfg.training.adam_beta1, cfg.training.adam_beta2),
        eps=cfg.training.adam_eps,
        weight_decay=cfg.training.weight_decay,
    )
    sched = get_scheduler("linear", opt, num_warmup_steps=warmup, num_training_steps=steps)
    history: list[dict[str, Any]] = []
    max_gn = 0.0
    final_loss = final_acc = spread = pmin = pmax = 0.0
    for step in range(1, steps + 1):
        bundle.model.train()
        opt.zero_grad()
        out = bundle.model(**enc, labels=labels)
        out.loss.backward()
        clip = cfg.training.grad_clip
        gn = float(torch.nn.utils.clip_grad_norm_(bundle.model.parameters(), clip))
        max_gn = max(max_gn, gn)
        step_loss = float(out.loss.detach())
        opt.step()
        sched.step()
        if step == 1 or step % 30 == 0 or step == steps:
            bundle.model.eval()
            with torch.no_grad():
                logits = bundle.model(**enc).logits
                probs = torch.softmax(logits, dim=-1)[:, POSITIVE_INDEX]
                final_acc = float((logits.argmax(-1) == labels).float().mean())
                pmin, pmax = float(probs.min()), float(probs.max())
                spread = pmax - pmin
            final_loss = step_loss
            history.append(
                {
                    "step": step,
                    "loss": round(final_loss, 6),
                    "grad_norm": round(gn, 4),
                    "accuracy": round(final_acc, 4),
                    "prob_min": round(pmin, 6),
                    "prob_max": round(pmax, 6),
                    "prob_spread": round(spread, 6),
                }
            )

    # pooled-representation variance across the batch (M5 smoking gun: ~0 under transformers 5.x)
    captured: dict[str, Any] = {}
    h = bundle.model.classifier.register_forward_hook(
        lambda m, i, o: captured.__setitem__("x", i[0].detach().float().cpu())
    )
    bundle.model.eval()
    with torch.no_grad():
        bundle.model(**enc)
    h.remove()
    pooled_std = float(captured["x"].std(dim=0).mean())

    return OverfitResult(
        transformers_version=tv,
        device=device,
        n_pairs=len(pairs),
        steps=steps,
        lr=lr,
        prior_ce=round(prior, 6),
        final_loss=round(final_loss, 6),
        final_accuracy=round(final_acc, 6),
        prob_min=round(pmin, 6),
        prob_max=round(pmax, 6),
        prob_spread=round(spread, 6),
        pooled_repr_std=round(pooled_std, 6),
        max_grad_norm=round(max_gn, 4),
        delta_head_norm=round(float((head.detach() - pre_head).norm()), 6),
        delta_backbone_norm=round(float((bb.detach() - pre_bb).norm()), 6),
        verdict=_verdict(final_acc, final_loss, spread),
        history=history,
    )


__all__ = [
    "TRANSFORMERS_MAX_EXCLUSIVE",
    "TRANSFORMERS_MIN",
    "OverfitResult",
    "assert_transformers_supported",
    "overfit_one_batch",
    "real_balanced_subset",
    "transformers_supported",
    "trivial_pairs",
]
