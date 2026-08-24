"""Load / save the DeBERTa-v3-base sequence-pair classifier. Lazy-imports transformers/torch.

Pinned revision, no ``trust_remote_code`` (``model_type: deberta-v2`` is a native class). The main
revision ships only ``pytorch_model.bin`` (no safetensors) so the BASE load sets
``use_safetensors=False``; the checkpoints WE save are safetensors (docs/THREAT-MODEL.md).
Head convention: index 1 = CACHE_EQUIVALENT (positive), matching ``predict.POSITIVE_INDEX``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polarity.train.config import TrainConfig
from polarity.types import Label


@dataclass
class Bundle:
    model: Any
    tokenizer: Any
    device: str


_ID2LABEL = {0: Label.NOT_CACHE_EQUIVALENT.value, 1: Label.CACHE_EQUIVALENT.value}
_LABEL2ID = {v: k for k, v in _ID2LABEL.items()}


def load_tokenizer(cfg: TrainConfig) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        cfg.model.hf_id, revision=cfg.model.tokenizer_revision, use_fast=True
    )


def load_base(cfg: TrainConfig, *, device: str = "cpu") -> Bundle:
    from transformers import AutoModelForSequenceClassification

    tok = load_tokenizer(cfg)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.hf_id,
        revision=cfg.model.revision,
        num_labels=cfg.model.num_labels,
        trust_remote_code=cfg.model.trust_remote_code,
        use_safetensors=False,  # main revision has only pytorch_model.bin
        id2label=_ID2LABEL,
        label2id=_LABEL2ID,
    )
    model.to(device)
    return Bundle(model=model, tokenizer=tok, device=device)


def load_checkpoint(checkpoint_dir: Path, cfg: TrainConfig, *, device: str = "cpu") -> Bundle:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_dir, trust_remote_code=cfg.model.trust_remote_code
    )
    model.to(device)
    return Bundle(model=model, tokenizer=tok, device=device)


def save_checkpoint(bundle: Bundle, checkpoint_dir: Path) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    bundle.model.save_pretrained(checkpoint_dir, safe_serialization=True)
    bundle.tokenizer.save_pretrained(checkpoint_dir)


def count_params(model: Any) -> int:
    return int(sum(p.numel() for p in model.parameters()))


__all__ = [
    "Bundle",
    "count_params",
    "load_base",
    "load_checkpoint",
    "load_tokenizer",
    "save_checkpoint",
]
