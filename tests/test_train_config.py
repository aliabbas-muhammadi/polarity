import copy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from polarity.train import config as tc

REAL_CONFIG = Path("configs/train/m3-deberta-v3-base.yaml")
XP005_CONFIG = Path("configs/train/m3-2-deberta-fp32.yaml")
XP006_CONFIG = Path("configs/train/m3-3-deberta-fixed.yaml")


def _base() -> dict:
    return copy.deepcopy(yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8")))


def test_real_config_loads_and_matches_manifests():
    cfg = tc.load_config(REAL_CONFIG)
    assert cfg.model.hf_id == "microsoft/deberta-v3-base"
    assert cfg.model.trust_remote_code is False
    # approved shas equal the committed manifests (the refuse-to-start gate is satisfied today)
    live = tc.assert_manifests_match(cfg)
    assert live["train"] == cfg.data.approved_sha256["train"]


def test_fingerprint_is_deterministic_and_hyperparameter_sensitive():
    cfg = tc.load_config(REAL_CONFIG)
    fp = tc.experiment_fingerprint(cfg)
    assert fp == tc.experiment_fingerprint(tc.load_config(REAL_CONFIG))
    bumped = cfg.model_copy(
        update={"training": cfg.training.model_copy(update={"learning_rate": 3e-5})}
    )
    assert tc.experiment_fingerprint(bumped) != fp
    # a display-only field must NOT change identity
    renamed = cfg.model_copy(update={"experiment_id": "renamed"})
    assert tc.experiment_fingerprint(renamed) == fp


def test_recipe_fingerprint_keeps_xp005_and_xp006_recipe_identity_stable():
    xp005 = tc.load_config(XP005_CONFIG)
    xp006 = tc.load_config(XP006_CONFIG)

    assert tc.recipe_fingerprint(xp005) == (
        "pl1:04857dd3f1522863cc9e99378569403ab76f72f5f64d7fc59abc98e38a125c17"
    )
    assert tc.recipe_fingerprint(xp006) == tc.recipe_fingerprint(xp005)
    assert tc.experiment_fingerprint(xp006) == tc.recipe_fingerprint(xp006)


def test_xp006_config_declares_exact_release_profile():
    cfg = tc.load_config(XP006_CONFIG)

    assert cfg.env.transformers == "==4.57.6"
    assert cfg.env.torch == ">=2.6,<2.7"


@pytest.mark.parametrize(
    ("dependency", "constraint"),
    [
        ("transformers", ">=4.44,<5"),
        ("torch", ">=2"),
    ],
)
def test_xp006_config_rejects_loosened_release_profile(dependency: str, constraint: str):
    raw = yaml.safe_load(XP006_CONFIG.read_text(encoding="utf-8"))
    raw["env"][dependency] = constraint

    with pytest.raises(ValidationError, match="xp-006 release profile requires"):
        tc.TrainConfig.model_validate(raw)


def test_manifest_mismatch_refuses_to_start():
    cfg = tc.load_config(REAL_CONFIG)
    bad = cfg.model_copy(
        update={
            "data": cfg.data.model_copy(
                update={"approved_sha256": {**cfg.data.approved_sha256, "train": "0" * 64}}
            )
        }
    )
    with pytest.raises(tc.FingerprintMismatch, match="train manifest sha256 changed"):
        tc.assert_manifests_match(bad)


def test_fp16_is_disallowed():
    d = _base()
    d["training"]["precision"] = "fp16"
    with pytest.raises(ValidationError, match="fp16 is disallowed"):
        tc.TrainConfig.model_validate(d)


def test_trust_remote_code_must_be_false():
    d = _base()
    d["model"]["trust_remote_code"] = True
    with pytest.raises(ValidationError, match="trust_remote_code must be False"):
        tc.TrainConfig.model_validate(d)


def test_pinned_revision_required():
    d = _base()
    d["model"]["revision"] = "  "
    with pytest.raises(ValidationError, match="must be pinned"):
        tc.TrainConfig.model_validate(d)


@pytest.mark.parametrize(
    "path",
    [
        "manifests/with-mnli/train.jsonl",
        "manifests/eval/nan-nli.jsonl",
        "data/snapshots/veritas-golden/x.jsonl",
    ],
)
def test_forbidden_dataset_paths_rejected(path):
    d = _base()
    d["data"]["train_manifest"] = path
    with pytest.raises(ValidationError):
        tc.TrainConfig.model_validate(d)


def test_allowed_sources_locked_to_primary():
    d = _base()
    d["data"]["allowed_sources"] = ["cannot", "paws-wiki", "mnli"]
    with pytest.raises(ValidationError, match="allowed_sources"):
        tc.TrainConfig.model_validate(d)


def test_selection_and_calibration_are_validation_only():
    d = _base()
    d["selection"]["split"] = "internal_test"
    with pytest.raises(ValidationError, match="VALIDATION only"):
        tc.TrainConfig.model_validate(d)
    d2 = _base()
    d2["calibration"]["fit_split"] = "internal_test"
    with pytest.raises(ValidationError, match="VALIDATION"):
        tc.TrainConfig.model_validate(d2)


def test_recipe_locked():
    d = _base()
    d["data"]["recipe"] = "primary-full-rebuild"
    with pytest.raises(ValidationError, match="new dataset version"):
        tc.TrainConfig.model_validate(d)
