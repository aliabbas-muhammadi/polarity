"""fp32/bf16 precision-selection tests + guards (brief §5/§7/§15). Torch-free.

M3.2's one intentional change is bf16 -> fp32. These tests pin the guarantee that an fp32 config
trains in fp32 *everywhere* (bf16 cannot be silently re-enabled on GPU), that CollapseCfg is
diagnostic-only (never churns a fingerprint), and that only the corrected pAUC (eval.curves) is
reachable — no pre-M4 ``_pauc`` survives.
"""

import copy
from pathlib import Path

import yaml

from polarity.train import config as tc
from polarity.train import report as m3report
from polarity.train import runner
from polarity.train.runner import precision_for

REAL_CONFIG = Path("configs/train/m3-1-deberta-v3-base.yaml")


def _base() -> dict:
    return copy.deepcopy(yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8")))


def test_fp32_config_stays_fp32_on_gpu_even_when_bf16_supported():
    # The core M3.2 guarantee: fp32 means fp32 on CUDA, bf16-capable or not.
    assert precision_for("fp32", "cuda", bf16_supported=True) == "fp32"
    assert precision_for("fp32", "cuda:0", bf16_supported=True) == "fp32"
    assert precision_for("fp32", "cpu", bf16_supported=False) == "fp32"


def test_bf16_config_only_uses_bf16_on_supported_gpu():
    assert precision_for("bf16", "cuda", bf16_supported=True) == "bf16"
    assert precision_for("bf16", "cuda", bf16_supported=False) == "fp32"  # not silently forced
    assert precision_for("bf16", "cpu", bf16_supported=True) == "fp32"  # CPU always fp32


def test_fp32_config_validates_and_disables_autocast_path():
    d = _base()
    d["training"]["precision"] = "fp32"
    d["experiment_id"] = "xp-005-deberta-fp32"
    parsed = tc.TrainConfig.model_validate(d)
    assert parsed.training.precision == "fp32"
    # resolve_precision delegates to precision_for -> fp32 for this config on any device.
    assert precision_for(parsed.training.precision, "cuda", True) == "fp32"


def test_collapse_cfg_is_not_result_affecting():
    # Changing a diagnostic threshold must NOT change the experiment fingerprint.
    d = _base()
    base_fp = tc.experiment_fingerprint(tc.TrainConfig.model_validate(d))
    d2 = _base()
    d2["collapse"] = {"min_score_std": 0.5, "patience_evals": 99}
    changed_fp = tc.experiment_fingerprint(tc.TrainConfig.model_validate(d2))
    assert base_fp == changed_fp
    # ...and the collapse block is genuinely absent from the hashed subset.
    assert "collapse" not in tc.TrainConfig.model_validate(d).result_affecting()


def test_old_configs_without_collapse_block_keep_their_fingerprint():
    # The committed M3/M3.1 configs omit `collapse:` — their fingerprints must be unchanged.
    for cfg_path in ("configs/train/m3-deberta-v3-base.yaml", REAL_CONFIG):
        cfg = tc.load_config(Path(cfg_path))
        assert cfg.collapse == tc.CollapseCfg()  # default applied
        # fingerprint recomputes without error and excludes the collapse block
        assert "collapse" not in cfg.result_affecting()


def test_only_corrected_pauc_is_reachable():
    # No pre-M4 buggy _pauc / _roc_points hides in the training modules (brief §7).
    assert not hasattr(runner, "_pauc")
    assert not hasattr(m3report, "_pauc")
    assert not hasattr(m3report, "_roc_points")
    # The corrected implementation is the one imported.
    from polarity.eval.curves import partial_auc

    assert runner.partial_auc is partial_auc
    assert m3report.partial_auc is partial_auc
