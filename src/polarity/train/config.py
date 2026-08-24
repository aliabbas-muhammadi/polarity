"""Immutable M3 training configuration + experiment fingerprint + the refuse-to-start gate.

This is the torch-free pre-flight layer (brief §18/§26). Loading, hashing, and the manifest /
source / revision guards run without the ``train`` extra, so the paid runner can verify — before
renting a GPU — that the committed manifests still match the approved fingerprint and that no
forbidden dataset (Veritas, MNLI, QQP, PAWS-QQP, ANLI, NaN-NLI, NevIR) has crept into the recipe.

The fingerprint hashes only result-affecting fields (model+revision, dataset hashes,
hyperparameters, seed, selection rule) — never display/cost/env fields (docs/EXPERIMENTS.md).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from polarity.fingerprint import content_hash, sha256_file

# The ONLY sources allowed in the primary recipe. Anything else is a dataset-version change.
ALLOWED_SOURCES = frozenset({"cannot", "paws-wiki"})
# Substrings that must never appear in a training-manifest path (defense in depth vs the guards
# already in materialize.py). Excluded corpora + the sealed set + the eval probes.
FORBIDDEN_PATH_TOKENS = (
    "veritas",
    "with-mnli",
    "mnli",
    "qqp",
    "anli",
    "nan-nli",
    "eval",
    "synthetic",
)


class ModelCfg(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    hf_id: str
    revision: str
    tokenizer_revision: str
    license: str
    params: str
    num_labels: int = 2
    trust_remote_code: bool = False

    @field_validator("revision", "tokenizer_revision")
    @classmethod
    def _pinned(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model/tokenizer revision must be pinned (non-empty)")
        return v

    @field_validator("trust_remote_code")
    @classmethod
    def _no_remote_code(cls, v: bool) -> bool:
        if v:
            raise ValueError("trust_remote_code must be False (docs/THREAT-MODEL.md)")
        return v

    @field_validator("num_labels")
    @classmethod
    def _binary(cls, v: int) -> int:
        if v != 2:
            raise ValueError("M3 is a 2-class head (CACHE_EQUIVALENT vs NOT)")
        return v


class DataCfg(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    recipe: str  # "primary-capped-A"
    train_manifest: str
    validation_manifest: str
    internal_test_manifest: str
    approved_sha256: dict[str, str]  # keys: train / validation / internal_test
    allowed_sources: list[str]

    @model_validator(mode="after")
    def _checks(self) -> DataCfg:
        if self.recipe != "primary-capped-A":
            raise ValueError(
                f"unexpected recipe {self.recipe!r}; a rebuild is a new dataset version"
            )
        if set(self.allowed_sources) != set(ALLOWED_SOURCES):
            raise ValueError(f"allowed_sources must be exactly {sorted(ALLOWED_SOURCES)}")
        if set(self.approved_sha256) != {"train", "validation", "internal_test"}:
            raise ValueError("approved_sha256 needs keys train/validation/internal_test")
        for label, path in self._manifests().items():
            low = path.lower()
            for tok in FORBIDDEN_PATH_TOKENS:
                if tok in low:
                    raise ValueError(
                        f"forbidden dataset token {tok!r} in {label} manifest path {path!r}"
                    )
            if not low.startswith("manifests/primary/"):
                raise ValueError(
                    f"{label} manifest must live under manifests/primary/, got {path!r}"
                )
        return self

    def _manifests(self) -> dict[str, str]:
        return {
            "train": self.train_manifest,
            "validation": self.validation_manifest,
            "internal_test": self.internal_test_manifest,
        }


class TrainingCfg(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    objective: str  # "binary_cross_entropy"
    max_length: int
    truncation: str  # "longest_first" (symmetric pair task)
    batch_size: int
    gradient_accumulation: int
    learning_rate: float
    adam_beta1: float
    adam_beta2: float
    adam_eps: float  # DeBERTaV3 Table 15 = 1e-6, NOT torch's 1e-8 default
    weight_decay: float
    warmup_ratio: float
    scheduler: str
    epochs: int
    max_steps: int | None
    grad_clip: float
    precision: str  # bf16 (GPU) | fp32 (CPU/smoke) | fp16 (disallowed for DeBERTa)
    seed: int
    eval_every_steps: int
    early_stopping_patience: int
    class_weights: bool = False

    @field_validator("truncation")
    @classmethod
    def _trunc(cls, v: str) -> str:
        if v != "longest_first":
            raise ValueError("pair task must truncate 'longest_first' (neither side privileged)")
        return v

    @field_validator("precision")
    @classmethod
    def _prec(cls, v: str) -> str:
        if v == "fp16":
            raise ValueError("fp16 is disallowed for DeBERTa (disentangled attention overflows)")
        if v not in {"bf16", "fp32"}:
            raise ValueError("precision must be bf16 (GPU) or fp32 (CPU/smoke)")
        return v

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.gradient_accumulation


class SelectionCfg(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: str  # "recall_at_fpr" | "pauc"
    fpr_budget: float  # 0.01
    tie_break: str  # "earliest_step"
    pauc_fpr_max: float  # 0.02 (pAUC band; also the stability cross-check)
    split: str = "validation"

    @field_validator("metric")
    @classmethod
    def _metric(cls, v: str) -> str:
        if v not in {"recall_at_fpr", "pauc"}:
            raise ValueError("selection.metric must be 'recall_at_fpr' or 'pauc'")
        return v

    @field_validator("split")
    @classmethod
    def _val_only(cls, v: str) -> str:
        if v != "validation":
            raise ValueError(
                "checkpoint selection is VALIDATION only (never internal_test/Veritas)"
            )
        return v


class CalibrationCfg(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidates: list[str]  # ["identity","temperature","platt","isotonic"]
    fit_split: str = "validation"
    isotonic_min_obs: int = 50

    @field_validator("fit_split")
    @classmethod
    def _val_only(cls, v: str) -> str:
        if v != "validation":
            raise ValueError(
                "calibration is fit on VALIDATION only, applied frozen to internal_test"
            )
        return v


class SuccessCfg(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rule: str  # "paired_uncertainty"
    nli_baseline_recall: float  # 0.073 at validation-selected tau; test FP=16/883 (1.81%)
    nli_baseline_recall_ci: list[float]  # [0.055, 0.094]
    nli_baseline_eqn_recall: float  # 0.052
    nli_baseline_eqn_n: int  # 58
    secondary_eqn_margin: float = 0.05  # equivalent-negation must beat baseline by this (small-n)
    guardrail_expl_fp_margin: float = 0.05  # explicit-negation FP may not rise by more than this
    bootstrap_n: int = 2000
    bootstrap_seed: int = 12345


class CollapseCfg(BaseModel):
    """Pre-registered collapse-detection thresholds (brief §4). Monitoring, not a hyperparameter.

    Deliberately NOT in ``result_affecting`` (docs/EXPERIMENTS.md) — changing a diagnostic
    threshold must never churn an experiment's fingerprint, and old configs that omit this block
    keep their exact hash. A model that collapses to a (near-)constant output learns the class
    prior and nothing input-dependent (docs/M4-REPORT.md); ``runner.py`` evaluates these on the
    VALIDATION scores at each checkpoint and hard-stops the paid run once collapse is *sustained*.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_score_std: float = 1e-3  # near-constant validation scores => no discrimination
    min_unique_scores: int = 2  # a single distinct score is a literal constant output
    min_class_separation: float = 1e-3  # |mean(pos) - mean(neg)| below this => no signal
    warmup_evals: int = 3  # do not judge collapse before this many validation checkpoints
    patience_evals: int = 3  # consecutive collapsed checkpoints (post-warmup) that trigger a stop


class CostCfg(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_runtime_hours: float
    max_estimated_cost_aud: float
    planning_hourly_rate_aud: float


class EnvCfg(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    python: str
    transformers: str
    torch: str
    hardware_class: str  # planning label; the real run records the true env to the run dir


class TrainConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str
    model: ModelCfg
    data: DataCfg
    training: TrainingCfg
    selection: SelectionCfg
    calibration: CalibrationCfg
    success: SuccessCfg
    cost: CostCfg
    env: EnvCfg
    collapse: CollapseCfg = Field(default_factory=CollapseCfg)

    @model_validator(mode="after")
    def xp006_release_profile_is_exact(self) -> TrainConfig:
        if self.experiment_id == "xp-006-deberta-transformers-fixed" and (
            self.env.transformers != "==4.57.6" or self.env.torch != ">=2.6,<2.7"
        ):
            raise ValueError(
                "xp-006 release profile requires transformers ==4.57.6 and torch >=2.6,<2.7"
            )
        return self

    def result_affecting(self) -> dict[str, Any]:
        """The subset the fingerprint hashes — result-affecting only (docs/EXPERIMENTS.md)."""
        return {
            "model": {
                "hf_id": self.model.hf_id,
                "revision": self.model.revision,
                "tokenizer_revision": self.model.tokenizer_revision,
                "num_labels": self.model.num_labels,
            },
            "data": {
                "recipe": self.data.recipe,
                "approved_sha256": self.data.approved_sha256,
                "allowed_sources": sorted(self.data.allowed_sources),
            },
            "training": self.training.model_dump(mode="json"),
            "selection": self.selection.model_dump(mode="json"),
        }


class FingerprintMismatch(RuntimeError):
    """Raised when the live manifests differ from the approved fingerprint (refuse to start)."""


def load_config(path: Path) -> TrainConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return TrainConfig.model_validate(raw)


def recipe_fingerprint(cfg: TrainConfig) -> str:
    """Stable identity for model/data/hyperparameters, independent of execution environment."""
    return content_hash(cfg.result_affecting())


def experiment_fingerprint(cfg: TrainConfig) -> str:
    """Compatibility alias for the pre-hardening recipe fingerprint name."""
    return recipe_fingerprint(cfg)


def live_manifest_hashes(cfg: TrainConfig, *, root: Path = Path(".")) -> dict[str, str]:
    return {
        "train": sha256_file(root / cfg.data.train_manifest),
        "validation": sha256_file(root / cfg.data.validation_manifest),
        "internal_test": sha256_file(root / cfg.data.internal_test_manifest),
    }


def assert_manifests_match(cfg: TrainConfig, *, root: Path = Path(".")) -> dict[str, str]:
    """Refuse-to-start gate: live manifest bytes must equal the approved shas. Returns them."""
    live = live_manifest_hashes(cfg, root=root)
    for key, approved in cfg.data.approved_sha256.items():
        if live[key] != approved:
            raise FingerprintMismatch(
                f"{key} manifest sha256 changed: live {live[key]} != approved {approved} "
                "— a dataset-version change requires review before training (brief §1/§4)."
            )
    return live


__all__ = [
    "ALLOWED_SOURCES",
    "CalibrationCfg",
    "CollapseCfg",
    "CostCfg",
    "DataCfg",
    "EnvCfg",
    "FingerprintMismatch",
    "ModelCfg",
    "SelectionCfg",
    "SuccessCfg",
    "TrainConfig",
    "TrainingCfg",
    "assert_manifests_match",
    "experiment_fingerprint",
    "live_manifest_hashes",
    "load_config",
    "recipe_fingerprint",
]
