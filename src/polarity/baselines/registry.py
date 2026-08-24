"""Pinned, license-cleared baseline models (M2). Like data/sources.py but for models.

All are downloaded free at a pinned HF revision, run on CPU, and require no trust_remote_code.
Excluded (documented in docs/M2-METHODOLOGY.md): MoritzLaurer NLI models (ANLI = CC-BY-NC
training data) and quora-roberta (Quora ToS). Label order differs per model, so the NLI scorer
resolves the entailment index from ``config.label2id`` rather than hardcoding it.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class ModelRole(StrEnum):
    EMBEDDING = "embedding"
    NLI_CROSS_ENCODER = "nli_cross_encoder"


class ModelSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    hf_id: str
    revision: str
    role: ModelRole
    license: str
    params: str
    pooling: str = ""
    normalize: bool = False
    prefix: str = ""  # symmetric-use prefix; "" = none (bge-small needs none)
    trust_remote_code: bool = False
    notes: str = ""


MODELS: dict[str, ModelSpec] = {
    "bge-small": ModelSpec(
        key="bge-small",
        hf_id="BAAI/bge-small-en-v1.5",
        revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        role=ModelRole.EMBEDDING,
        license="MIT",
        params="~33M / 384-dim",
        pooling="cls",
        normalize=True,
        prefix="",
        notes="no instruction prefix for symmetric similarity",
    ),
    "nli-deberta-v3-base": ModelSpec(
        key="nli-deberta-v3-base",
        hf_id="cross-encoder/nli-deberta-v3-base",
        revision="6c749ce3425cd33b46d187e45b92bbf96ee12ec7",
        role=ModelRole.NLI_CROSS_ENCODER,
        license="Apache-2.0",
        params="184M (DebertaV2)",
        notes="trained SNLI+MNLI (no ANLI); entailment index via config.label2id",
    ),
}


def get_model(key: str) -> ModelSpec:
    if key not in MODELS:
        raise KeyError(f"unknown model {key!r} (known: {sorted(MODELS)})")
    return MODELS[key]


__all__ = ["MODELS", "ModelRole", "ModelSpec", "get_model"]
