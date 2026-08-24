import copy
import json
from pathlib import Path

import pytest
import yaml

from polarity.baselines.schema import BaselinePrediction
from polarity.train import report as m3
from polarity.train.config import load_config
from polarity.types import Category, Label

REAL_CONFIG = Path("configs/train/m3-deberta-v3-base.yaml")
EXP = "test-xp"

# pair_id -> (gold, category); shared across the M3 model and the NLI baseline.
_PAIRS = [
    ("p1", "CACHE_EQUIVALENT", "paraphrase"),
    ("p2", "CACHE_EQUIVALENT", "paraphrase"),
    ("p3", "CACHE_EQUIVALENT", "equivalent_negation"),
    ("p4", "CACHE_EQUIVALENT", "equivalent_negation"),
    ("p5", "CACHE_EQUIVALENT", "hard_paraphrase"),
    ("p6", "CACHE_EQUIVALENT", "paraphrase"),
    ("n1", "NOT_CACHE_EQUIVALENT", "explicit_negation"),
    ("n2", "NOT_CACHE_EQUIVALENT", "explicit_negation"),
    ("n3", "NOT_CACHE_EQUIVALENT", "explicit_negation"),
    ("n4", "NOT_CACHE_EQUIVALENT", "unknown"),
    ("n5", "NOT_CACHE_EQUIVALENT", "unknown"),
    ("n6", "NOT_CACHE_EQUIVALENT", "lexical_contradiction"),
]
_M3 = {
    "p1": 0.95,
    "p2": 0.92,
    "p3": 0.9,
    "p4": 0.88,
    "p5": 0.8,
    "p6": 0.78,
    "n1": 0.2,
    "n2": 0.15,
    "n3": 0.1,
    "n4": 0.25,
    "n5": 0.05,
    "n6": 0.3,
}
_NLI = {
    "p1": 0.55,
    "p2": 0.5,
    "p3": 0.45,
    "p4": 0.4,
    "p5": 0.35,
    "p6": 0.3,
    "n1": 0.28,
    "n2": 0.3,
    "n3": 0.2,
    "n4": 0.32,
    "n5": 0.15,
    "n6": 0.34,
}


def _preds(scores: dict[str, float]) -> list[BaselinePrediction]:
    return [
        BaselinePrediction(
            pair_id=pid,
            gold=Label(gold),
            score=scores[pid],
            category=Category(cat),
            source="cannot",
            has_neg_cue=("negation" in cat),
            token_overlap=0.5,
            len_diff=0,
        )
        for pid, gold, cat in _PAIRS
    ]


def _write(root: Path, rel: Path, preds: list[BaselinePrediction]) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(p.model_dump_json() for p in preds) + "\n", encoding="utf-8")


def _fixture(root: Path) -> Path:
    cfg = copy.deepcopy(yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8")))
    cfg["experiment_id"] = EXP
    cfg["success"]["bootstrap_n"] = 200  # fast + deterministic
    cfg_path = Path("config.yaml")
    (root / cfg_path).write_text(yaml.safe_dump(cfg), encoding="utf-8")
    for split in ("validation", "internal_test"):
        _write(root, Path("predictions/m3") / EXP / f"{split}.jsonl", _preds(_M3))
        _write(root, Path("predictions/m2/nli-cross-encoder") / f"{split}.jsonl", _preds(_NLI))
    return cfg_path


def test_build_and_check_reproduce(tmp_path: Path):
    cfg_path = _fixture(tmp_path)
    files = m3.build_all(cfg_path, out_root=tmp_path)
    assert f"metrics-{EXP}.json" in files and f"disposition-{EXP}.json" in files
    assert m3.check_all(cfg_path, out_root=tmp_path) == []


def test_drift_detected(tmp_path: Path):
    cfg_path = _fixture(tmp_path)
    m3.build_all(cfg_path, out_root=tmp_path)
    (tmp_path / f"reports/m3/disposition-{EXP}.json").write_text("{}\n", encoding="utf-8")
    assert f"disposition-{EXP}.json" in m3.check_all(cfg_path, out_root=tmp_path)


def test_deterministic_rebuild(tmp_path: Path):
    cfg_path = _fixture(tmp_path)
    m3.build_all(cfg_path, out_root=tmp_path)
    first = (tmp_path / f"reports/m3/metrics-{EXP}.json").read_text()
    m3.build_all(cfg_path, out_root=tmp_path)
    assert first == (tmp_path / f"reports/m3/metrics-{EXP}.json").read_text()


def test_xp005_calibration_params_use_report_precision():
    cfg = load_config(Path("configs/train/m3-2-deberta-fp32.yaml"))
    val = m3.load(cfg.experiment_id, "validation", Path("."))
    test = m3.load(cfg.experiment_id, "internal_test", Path("."))

    calibration = m3._calibration(val, test, cfg)

    assert calibration["selected_method"] == "platt"
    assert calibration["selected_params"] == {"a": 100.500884, "b": 113.889939}


def test_report_has_negation_slices_and_paired_comparison(tmp_path: Path):
    cfg_path = _fixture(tmp_path)
    m3.build_all(cfg_path, out_root=tmp_path)
    rep = json.loads((tmp_path / f"reports/m3/metrics-{EXP}.json").read_text())
    assert rep["equivalent_negation"]["m3"]["n_positive"] == 2
    assert rep["explicit_negation"]["m3"]["n_negative"] == 3
    assert "recall_delta_bootstrap" in rep["paired_vs_nli_baseline"]
    paired = rep["paired_vs_nli_baseline"]
    assert paired["selection"] == "each model's threshold selected on validation to target FPR<=1%"
    assert paired["nli_baseline_internal_test_at_selected_threshold"]["fp"] == 0
    assert paired["nli_baseline_internal_test_at_selected_threshold"]["fpr"] == 0.0
    assert "nli_baseline_recall_reference" not in paired
    assert rep["disposition"]["suggested_disposition"] in {
        "STRONG_SUCCESS",
        "PARTIAL_SUCCESS",
        "NO_MATERIAL_GAIN",
        "REGRESSION",
    }
    assert rep["calibration"]["selected_method"] in {"identity", "temperature", "platt", "isotonic"}


def _paired_stats(
    a: list[BaselinePrediction], b: list[BaselinePrediction]
) -> tuple[dict[str, object], dict[str, object]]:
    bootstrap = m3._paired_recall_bootstrap(a, b, 0.5, 0.5, n_boot=20, seed=1)
    mcnemar = m3._mcnemar(a, b, 0.5, 0.5)
    return bootstrap, mcnemar


def _assert_paired_stats_reject(
    a: list[BaselinePrediction], b: list[BaselinePrediction], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        m3._paired_recall_bootstrap(a, b, 0.5, 0.5, n_boot=20, seed=1)
    with pytest.raises(ValueError, match=match):
        m3._mcnemar(a, b, 0.5, 0.5)


def test_paired_statistics_accept_exact_alignment():
    a = _preds(_M3)
    b = _preds(_NLI)

    bootstrap, mcnemar = _paired_stats(a, b)

    assert bootstrap["n_positive"] == 6
    assert "m3_correct_nli_wrong" in mcnemar


def test_paired_statistics_reject_missing_ids():
    a = _preds(_M3)
    b = _preds(_NLI)[:-1]

    _assert_paired_stats_reject(a, b, "prediction count mismatch")


def test_paired_statistics_reject_reordered_ids():
    a = _preds(_M3)
    b = _preds(_NLI)
    b[0], b[1] = b[1], b[0]

    _assert_paired_stats_reject(a, b, "pair_id mismatch at position 0")


def test_paired_statistics_reject_label_mismatch():
    a = _preds(_M3)
    b = _preds(_NLI)
    b[0] = b[0].model_copy(update={"gold": Label.NOT_CACHE_EQUIVALENT})

    _assert_paired_stats_reject(a, b, "gold label mismatch for pair_id p1")


def test_paired_statistics_reject_category_mismatch():
    a = _preds(_M3)
    b = _preds(_NLI)
    b[0] = b[0].model_copy(update={"category": Category.HARD_PARAPHRASE})

    _assert_paired_stats_reject(a, b, "category mismatch for pair_id p1")


def test_paired_statistics_reject_source_mismatch():
    a = _preds(_M3)
    b = _preds(_NLI)
    b[0] = b[0].model_copy(update={"source": "paws"})

    _assert_paired_stats_reject(a, b, "source mismatch for pair_id p1")


def test_training_health_block_flags_collapse_keylessly(tmp_path: Path):
    # The healthy fixture model separates the classes -> not collapsed; the NLI baseline (also
    # separable here) -> not collapsed. This is the keyless mechanics gate (brief §4).
    cfg_path = _fixture(tmp_path)
    m3.build_all(cfg_path, out_root=tmp_path)
    rep = json.loads((tmp_path / f"reports/m3/metrics-{EXP}.json").read_text())
    th = rep["training_health"]
    assert th["internal_test"]["collapse"]["collapsed"] is False
    assert th["internal_test"]["class_separation"] > 0.1
    assert "min_score_std" in th["collapse_thresholds"]


def test_fp_direction_high_scored_negative_counts_as_fp(tmp_path: Path):
    # a NOT-labelled pair scored above threshold is a false positive at that operating point
    cfg_path = _fixture(tmp_path)
    m3.build_all(cfg_path, out_root=tmp_path)
    rep = json.loads((tmp_path / f"reports/m3/metrics-{EXP}.json").read_text())
    op = rep["primary_operating_point"]["internal_test"]
    assert op["fp"] == sum(
        1 for pid, gold, _ in _PAIRS if gold == "NOT_CACHE_EQUIVALENT" and _M3[pid] >= op["tau"]
    )


def _base_disp_inputs():
    success = load_config(REAL_CONFIG).success
    m3_recall_ci = {"ci_low": 0.2, "ci_high": 0.5}
    paired = {"ci_low": 0.1, "ci_high": 0.3}
    m3_eqn = {"recall_at_op": 0.3, "fp_rate_at_op": None}
    nli_eqn = {"recall_at_op": 0.05, "fp_rate_at_op": None}
    m3_expl = {"fp_rate_at_op": 0.0}
    nli_expl = {"fp_rate_at_op": 0.0}
    return success, m3_recall_ci, paired, m3_eqn, nli_eqn, m3_expl, nli_expl


def test_disposition_strong():
    d = m3._disposition(*_base_disp_inputs())
    assert d["suggested_disposition"] == "STRONG_SUCCESS"


def test_disposition_partial_when_secondary_fails():
    s, rci, paired, m3e, nlie, m3x, nlix = _base_disp_inputs()
    m3e = {"recall_at_op": 0.06, "fp_rate_at_op": None}  # < 0.052 + 0.05
    d = m3._disposition(s, rci, paired, m3e, nlie, m3x, nlix)
    assert d["suggested_disposition"] == "PARTIAL_SUCCESS"


def test_disposition_no_material_when_primary_fails():
    s, rci, _, m3e, nlie, m3x, nlix = _base_disp_inputs()
    rci = {"ci_low": 0.02, "ci_high": 0.2}  # < baseline 0.073
    paired = {"ci_low": 0.0, "ci_high": 0.2}  # includes 0
    m3e = {"recall_at_op": 0.06, "fp_rate_at_op": None}
    d = m3._disposition(s, rci, paired, m3e, nlie, m3x, nlix)
    assert d["suggested_disposition"] == "NO_MATERIAL_GAIN"


def test_disposition_regression_on_guardrail_breach():
    s, rci, paired, m3e, nlie, _, nlix = _base_disp_inputs()
    m3x = {"fp_rate_at_op": 0.5}  # explicit-negation unsafe FP jumps
    d = m3._disposition(s, rci, paired, m3e, nlie, m3x, nlix)
    assert d["suggested_disposition"] == "REGRESSION"
