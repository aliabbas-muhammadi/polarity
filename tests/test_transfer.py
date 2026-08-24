"""M6 transfer tests (brief §27). Torch-free / offline — the frozen protocol derivation, the
calibrator round-trip, per-pair-type slices, operating-point metrics, and reproduction
classification. Scoring the checkpoint is author-run and lives in `polarity transfer veritas`.
"""

import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from polarity.artifact import ArtifactIdentity, load_artifact_identity
from polarity.baselines.schema import BaselinePrediction
from polarity.cli import app
from polarity.transfer import (
    FrozenProtocol,
    GoldenScore,
    _calibrator,
    build_report_files,
    check_report,
    derive_protocol,
    dump,
    load_preds,
    load_scores,
    reproduction_classify,
    transfer_report,
)
from polarity.types import Category, Label


def _pred(pid: str, gold: str, score: float) -> BaselinePrediction:
    return BaselinePrediction(
        pair_id=pid,
        gold=Label(gold),
        score=score,
        category=Category.PARAPHRASE,
        source="x",
        has_neg_cue=False,
        token_overlap=0.5,
        len_diff=0,
    )


def _val_preds() -> list[BaselinePrediction]:
    # 30 positives high, 30 negatives low — separable, miscalibrated toward the extremes.
    pos = [_pred(f"p{i}", "CACHE_EQUIVALENT", 0.6 + 0.01 * (i % 20)) for i in range(30)]
    neg = [_pred(f"n{i}", "NOT_CACHE_EQUIVALENT", 0.05 + 0.01 * (i % 20)) for i in range(30)]
    return pos + neg


def _gs(pid: str, pt: str, equiv: bool, raw: float) -> GoldenScore:
    return GoldenScore(
        id=pid, group=pid, pair_type=pt, gold_equivalent=equiv, raw_score=raw, calibrated=raw
    )


def _artifact_identity() -> ArtifactIdentity:
    return ArtifactIdentity(
        checkpoint_sha256="a" * 64,
        artifact_bundle_fingerprint=f"pl1:{'b' * 64}",
        recipe_fingerprint=f"pl1:{'c' * 64}",
        execution_fingerprint=f"pl1:{'d' * 64}",
    )


def _proto(tau: float = 0.5, *, experiment_id: str = "xp-test") -> FrozenProtocol:
    observed = {"tp": 1, "fp": 0, "tn": 1, "fn": 0, "recall": 1.0, "fpr": 0.0}
    return FrozenProtocol.model_validate(
        {
            "protocol_version": 2,
            "experiment_id": experiment_id,
            "model_fingerprint": f"pl1:{'c' * 64}",
            "checkpoint_sha256": "a" * 64,
            "artifact_bundle_fingerprint": f"pl1:{'b' * 64}",
            "recipe_fingerprint": f"pl1:{'c' * 64}",
            "execution_fingerprint": f"pl1:{'d' * 64}",
            "calibrator_method": "identity",
            "calibrator_params": {},
            "operating_points": {
                "zero_fp": {
                    "selection_split": "validation",
                    "selection_rule": "zero_observed_fp",
                    "target_fpr": None,
                    "threshold": tau,
                    "observed_selection": observed,
                },
                "fpr_le_0.01": {
                    "selection_split": "validation",
                    "selection_rule": "fpr_budget",
                    "target_fpr": 0.01,
                    "threshold": tau,
                    "observed_selection": observed,
                },
                "fpr_le_0.05": {
                    "selection_split": "validation",
                    "selection_rule": "fpr_budget",
                    "target_fpr": 0.05,
                    "threshold": tau,
                    "observed_selection": observed,
                },
            },
            "validation_n": 2,
        }
    )


def _write_proto(tmp_path: Path, tau: float = 0.5) -> Path:
    path = tmp_path / "protocol.json"
    path.write_text(
        dump(_proto(tau, experiment_id="xp-cli").model_dump(mode="json")), encoding="utf-8"
    )
    return path


def test_derive_protocol_freezes_thresholds_and_calibrator():
    proto = derive_protocol(
        _val_preds(),
        experiment_id="xp-test",
        artifact_identity=_artifact_identity(),
        isotonic_min_obs=10,
    )
    assert set(proto.thresholds) == {"zero_fp", "fpr_le_0.01", "fpr_le_0.05"}
    assert proto.calibrator_method == "isotonic"
    # deterministic
    proto2 = derive_protocol(
        _val_preds(),
        experiment_id="xp-test",
        artifact_identity=_artifact_identity(),
        isotonic_min_obs=10,
    )
    assert proto.model_dump() == proto2.model_dump()


def test_zero_fp_threshold_survives_protocol_json_roundtrip_without_admitting_top_negative():
    val = _val_preds()
    top_negative = 0.543210987654321
    val.append(_pred("top-negative", "NOT_CACHE_EQUIVALENT", top_negative))
    proto = derive_protocol(
        val,
        experiment_id="xp-test",
        artifact_identity=_artifact_identity(),
        isotonic_min_obs=10,
    )

    assert proto.protocol_version == 2
    assert proto.operating_points["zero_fp"].threshold == math.nextafter(top_negative, 2.0)

    restored = FrozenProtocol.model_validate_json(
        json.dumps(proto.model_dump(mode="json"), sort_keys=True)
    )
    assert restored.operating_points["zero_fp"].threshold > top_negative


def test_protocol_v2_rejects_operating_point_key_budget_mismatch():
    proto = derive_protocol(
        _val_preds(),
        experiment_id="xp-test",
        artifact_identity=_artifact_identity(),
        isotonic_min_obs=10,
    )
    payload = proto.model_dump(mode="json")
    payload["operating_points"]["fpr_le_0.01"]["target_fpr"] = 0.05

    with pytest.raises(ValidationError, match=r"fpr_le_0\.01"):
        FrozenProtocol.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "malformed"),
    [
        ("checkpoint_sha256", "deadbeef"),
        ("artifact_bundle_fingerprint", "pl1:bundle"),
        ("recipe_fingerprint", "pl1:recipe"),
        ("execution_fingerprint", "pl1:execution"),
    ],
)
def test_protocol_v2_rejects_malformed_artifact_identity(field: str, malformed: str):
    payload = _proto().model_dump(mode="json")
    payload[field] = malformed
    if field == "recipe_fingerprint":
        payload["model_fingerprint"] = malformed

    with pytest.raises(ValidationError):
        FrozenProtocol.model_validate(payload)


def test_protocol_v2_rejects_inconsistent_selection_class_counts():
    payload = _proto().model_dump(mode="json")
    payload["operating_points"]["fpr_le_0.05"]["observed_selection"] = {
        "tp": 0,
        "fp": 0,
        "tn": 2,
        "fn": 0,
        "recall": 0.0,
        "fpr": 0.0,
    }

    with pytest.raises(ValidationError, match="class counts"):
        FrozenProtocol.model_validate(payload)


def test_protocol_v2_budget_validation_uses_counts_not_rounded_descriptive_fpr():
    payload = _proto().model_dump(mode="json")
    payload["validation_n"] = 10_000_001
    zero_fp = {
        "tp": 1,
        "fp": 0,
        "tn": 10_000_000,
        "fn": 0,
        "recall": 1.0,
        "fpr": 0.0,
    }
    rounded_down = {
        "tp": 1,
        "fp": 100_001,
        "tn": 9_899_999,
        "fn": 0,
        "recall": 1.0,
        "fpr": 0.01,
    }
    payload["operating_points"]["zero_fp"]["observed_selection"] = zero_fp
    payload["operating_points"]["fpr_le_0.01"]["observed_selection"] = rounded_down
    payload["operating_points"]["fpr_le_0.05"]["observed_selection"] = rounded_down

    with pytest.raises(ValidationError, match="observed selection FPR exceeds target_fpr"):
        FrozenProtocol.model_validate(payload)


def test_calibrator_roundtrip_is_not_a_silent_identity():
    # The isotonic breakpoints must survive serialization (regression: they live in private fields).
    proto = derive_protocol(
        _val_preds(),
        experiment_id="xp-test",
        artifact_identity=_artifact_identity(),
        isotonic_min_obs=10,
    )
    cal = _calibrator(proto)
    out = cal.apply([0.1, 0.9])
    assert out[0] < out[1]  # monotone, separating
    assert cal.apply([0.9]) != [0.9] or cal.apply([0.1]) != [0.1]  # not identity


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_method",
        "unexpected_key",
        "breakpoint_count",
        "non_finite",
        "unordered_x",
        "unordered_y",
    ],
)
def test_protocol_v2_rejects_malformed_isotonic_calibrator_contract(mutation: str):
    payload = derive_protocol(
        _val_preds(),
        experiment_id="xp-test",
        artifact_identity=_artifact_identity(),
        isotonic_min_obs=10,
    ).model_dump(mode="json")
    params = payload["calibrator_params"]
    if mutation == "unknown_method":
        payload["calibrator_method"] = "mystery"
    elif mutation == "unexpected_key":
        params["unexpected"] = 1
    elif mutation == "breakpoint_count":
        params["breakpoints"] += 1
    elif mutation == "non_finite":
        params["iso_y"][0] = math.nan
    elif mutation == "unordered_x":
        params["iso_x"][1] = params["iso_x"][0]
    elif mutation == "unordered_y":
        params["iso_y"][1] = math.nextafter(params["iso_y"][0], -math.inf)
    else:
        raise AssertionError(f"unhandled calibrator mutation: {mutation}")

    with pytest.raises(ValidationError, match="calibrator"):
        FrozenProtocol.model_validate(payload)


@pytest.mark.parametrize(
    ("method", "params"),
    [
        ("identity", {"unexpected": 1}),
        ("temperature", {"temperature": 0.0}),
        ("platt", {"a": 1.0}),
    ],
)
def test_protocol_v2_rejects_invalid_non_isotonic_calibrator_contracts(
    method: str, params: dict[str, float]
):
    payload = _proto().model_dump(mode="json")
    payload["calibrator_method"] = method
    payload["calibrator_params"] = params

    with pytest.raises(ValidationError, match="calibrator"):
        FrozenProtocol.model_validate(payload)


def test_transfer_report_operating_points_and_slices():
    # 2 equivalent (should hit), 2 negation negatives (must not hit), clearly separated.
    scores = [
        _gs("e1", "paraphrase-should-hit", True, 0.95),
        _gs("e2", "paraphrase-should-hit", True, 0.90),
        _gs("x1", "negation-must-not-hit", False, 0.10),
        _gs("x2", "negation-must-not-hit", False, 0.05),
    ]
    proto = _proto()
    rep = transfer_report(scores, proto)
    op = rep["operating_points"]["zero_fp"]
    assert op["realized"]["overall"]["recall"] == 1.0
    assert op["realized"]["overall"]["false_positives"] == 0
    slc = op["realized"]["by_pair_type"]
    assert slc["paraphrase-should-hit"]["recall"] == 1.0
    assert slc["negation-must-not-hit"]["false_positives"] == 0
    assert rep["n_pairs"] == 4 and rep["n_equivalent"] == 2


def test_transfer_report_separates_selection_budget_from_realized_and_threshold_free_rates():
    scores = [
        _gs("e1", "paraphrase-should-hit", True, 0.95),
        _gs("e2", "paraphrase-should-hit", True, 0.90),
        _gs("x1", "different-answer-must-not-hit", False, 0.91),
        _gs("x2", "negation-must-not-hit", False, 0.05),
    ]

    rep = transfer_report(scores, _proto())
    selected = rep["operating_points"]["fpr_le_0.01"]
    assert selected["selection"]["target_fpr"] == 0.01
    assert selected["realized"]["split"] == "veritas"
    assert selected["realized"]["overall"]["fpr"] == 0.5
    assert selected["realized"]["overall"]["fp"] == 1

    swept = rep["threshold_free"]["recall_at_fpr_le_0.01"]
    assert swept["split"] == "veritas"
    assert swept["overall"]["recall"] == 0.5
    assert swept["overall"]["fpr"] == 0.0


def test_committed_veritas_report_has_correct_selection_and_realized_semantics():
    protocol_path = Path("reports/m6/protocol-xp-006-deberta-transformers-fixed.json")
    scores_path = Path("predictions/m6/veritas/xp-006-deberta-transformers-fixed.jsonl")
    validation_path = Path("predictions/m3/xp-006-deberta-transformers-fixed/validation.jsonl")
    proto = FrozenProtocol.model_validate_json(protocol_path.read_text(encoding="utf-8"))
    scores = load_scores(scores_path)

    top_validation_negative = max(
        p.score for p in load_preds(validation_path) if p.gold == Label.NOT_CACHE_EQUIVALENT
    )
    assert proto.operating_points["zero_fp"].threshold == 0.9996990000000001
    assert proto.operating_points["zero_fp"].threshold > top_validation_negative

    rep = transfer_report(scores, proto)
    primary = rep["operating_points"]["fpr_le_0.01"]
    assert primary["selection"]["selection_split"] == "validation"
    assert primary["selection"]["threshold"] == 0.994833
    assert primary["selection"]["observed_selection"] == {
        "tp": 707,
        "fp": 8,
        "tn": 855,
        "fn": 74,
        "recall": 0.90525,
        "fpr": 0.00927,
    }
    assert primary["realized"]["overall"]["tp"] == 31
    assert primary["realized"]["overall"]["fp"] == 2
    assert primary["realized"]["overall"]["fpr"] == 0.045455
    negation = primary["realized"]["by_pair_type"]["negation-must-not-hit"]
    assert negation["false_positives"] == 0 and negation["n_negative"] == 22

    swept = rep["threshold_free"]["recall_at_fpr_le_0.01"]["overall"]
    assert swept["tp"] == 24 and swept["fp"] == 0
    assert swept["recall"] == 0.774194 and swept["fpr"] == 0.0
    assert check_report(scores, proto) == []


def test_committed_m6_protocol_and_report_bind_full_artifact_identity():
    identity = load_artifact_identity(
        Path("reports/m6/model-manifest-xp-006-deberta-transformers-fixed.json")
    )
    protocol = json.loads(
        Path("reports/m6/protocol-xp-006-deberta-transformers-fixed.json").read_text(
            encoding="utf-8"
        )
    )
    report = json.loads(
        Path("reports/m6/transfer-xp-006-deberta-transformers-fixed.json").read_text(
            encoding="utf-8"
        )
    )

    for artifact in (protocol, report):
        assert artifact["checkpoint_sha256"] == identity.checkpoint_sha256
        assert artifact["artifact_bundle_fingerprint"] == identity.artifact_bundle_fingerprint
        assert artifact["recipe_fingerprint"] == identity.recipe_fingerprint
        assert artifact["execution_fingerprint"] == identity.execution_fingerprint
        assert artifact["model_fingerprint"] == identity.recipe_fingerprint


def test_check_report_detects_corrupt_or_missing_generated_report(tmp_path: Path):
    protocol_path = Path("reports/m6/protocol-xp-006-deberta-transformers-fixed.json")
    scores_path = Path("predictions/m6/veritas/xp-006-deberta-transformers-fixed.jsonl")
    proto = FrozenProtocol.model_validate_json(protocol_path.read_text(encoding="utf-8"))
    scores = load_scores(scores_path)
    files = build_report_files(scores, proto)
    assert len(files) == 1
    fname, expected = next(iter(files.items()))
    report_path = tmp_path / "reports/m6" / fname
    report_path.parent.mkdir(parents=True)
    report_path.write_text(expected, encoding="utf-8")
    assert check_report(scores, proto, tmp_path) == []

    report_path.write_text("{}\n", encoding="utf-8")
    assert check_report(scores, proto, tmp_path) == [fname]

    report_path.unlink()
    assert check_report(scores, proto, tmp_path) == [fname]


def test_transfer_flags_unsafe_false_positive():
    # a negation negative scored above threshold is a false positive (the dangerous failure)
    scores = [
        _gs("e1", "paraphrase-should-hit", True, 0.9),
        _gs("x1", "negation-must-not-hit", False, 0.8),  # unsafe FP at tau 0.5
    ]
    proto = _proto()
    rep = transfer_report(scores, proto)
    assert rep["operating_points"]["zero_fp"]["realized"]["overall"]["false_positives"] == 1


def test_reproduction_classify_verdicts():
    committed = [_pred("a", "CACHE_EQUIVALENT", 0.9), _pred("b", "NOT_CACHE_EQUIVALENT", 0.1)]
    same = [_pred("a", "CACHE_EQUIVALENT", 0.9), _pred("b", "NOT_CACHE_EQUIVALENT", 0.1)]
    agreement = reproduction_classify(same, committed, protocol=_proto())
    assert agreement["verdict"] == "SCORE_AGREEMENT_WITHIN_TOLERANCE"
    assert agreement["score_agreement_tolerance"] == 1e-6
    assert agreement["score_agreement_rule"] == "aligned per-row absolute score difference < 1e-6"
    assert "EXACT_" + "REPRODUCTION" not in json.dumps(agreement)
    noisy = [_pred("a", "CACHE_EQUIVALENT", 0.905), _pred("b", "NOT_CACHE_EQUIVALENT", 0.098)]
    assert (
        reproduction_classify(noisy, committed, protocol=_proto())["verdict"]
        == "METRIC_REPRODUCTION_WITH_NUMERICAL_VARIATION"
    )
    drifted = [_pred("a", "CACHE_EQUIVALENT", 0.2), _pred("b", "NOT_CACHE_EQUIVALENT", 0.8)]
    assert (
        reproduction_classify(drifted, committed, protocol=_proto())["verdict"]
        == "MATERIAL_REPRODUCTION_DRIFT"
    )
    assert (
        reproduction_classify([_pred("a", "CACHE_EQUIVALENT", 0.9)], committed, protocol=_proto())[
            "verdict"
        ]
        == "MATERIAL_REPRODUCTION_DRIFT"
    )  # id set differs


def test_score_agreement_preserves_sub_micro_score_difference_in_report():
    committed = [_pred("a", "CACHE_EQUIVALENT", 0.5)]
    candidate = [_pred("a", "CACHE_EQUIVALENT", 0.5000009)]

    result = reproduction_classify(candidate, committed, protocol=_proto())

    assert result["verdict"] == "SCORE_AGREEMENT_WITHIN_TOLERANCE"
    assert result["max_abs_score_diff"] == abs(candidate[0].score - committed[0].score)
    assert result["max_abs_score_diff"] < 1e-6


@pytest.mark.parametrize(
    ("delta", "expected_verdict"),
    [
        (
            math.nextafter(1e-6, 0.0),
            "SCORE_AGREEMENT_WITHIN_TOLERANCE",
        ),
        (1e-6, "METRIC_REPRODUCTION_WITH_NUMERICAL_VARIATION"),
        (
            math.nextafter(1e-6, math.inf),
            "METRIC_REPRODUCTION_WITH_NUMERICAL_VARIATION",
        ),
    ],
)
def test_score_agreement_boundary_is_preserved_in_serialized_diagnostic(
    delta: float, expected_verdict: str
):
    committed = [_pred("a", "NOT_CACHE_EQUIVALENT", 0.0)]
    candidate = [_pred("a", "NOT_CACHE_EQUIVALENT", delta)]

    result = reproduction_classify(candidate, committed, protocol=_proto())
    serialized = json.loads(dump(result))

    assert result["verdict"] == expected_verdict
    assert result["max_abs_score_diff"] == delta
    assert serialized["max_abs_score_diff"] == delta


@pytest.mark.parametrize(
    ("delta", "expected_verdict"),
    [
        (
            math.nextafter(0.02, 0.0),
            "METRIC_REPRODUCTION_WITH_NUMERICAL_VARIATION",
        ),
        (0.02, "METRIC_REPRODUCTION_WITH_NUMERICAL_VARIATION"),
        (math.nextafter(0.02, math.inf), "MATERIAL_REPRODUCTION_DRIFT"),
    ],
)
def test_score_drift_boundary_is_preserved_in_serialized_diagnostic(
    delta: float, expected_verdict: str
):
    committed = [_pred("a", "NOT_CACHE_EQUIVALENT", 0.0)]
    candidate = [_pred("a", "NOT_CACHE_EQUIVALENT", delta)]

    result = reproduction_classify(candidate, committed, protocol=_proto())
    serialized = json.loads(dump(result))

    assert result["verdict"] == expected_verdict
    assert result["max_abs_score_diff"] == delta
    assert serialized["max_abs_score_diff"] == delta


def test_recall_diagnostic_preserves_full_verdict_precision():
    committed = [
        _pred("p1", "CACHE_EQUIVALENT", 0.5),
        _pred("p2", "CACHE_EQUIVALENT", 0.5),
        _pred("p3", "CACHE_EQUIVALENT", 0.5),
        _pred("n1", "NOT_CACHE_EQUIVALENT", 0.49),
    ]
    candidate = [committed[0].model_copy(update={"score": 0.499}), *committed[1:]]

    result = reproduction_classify(candidate, committed, protocol=_proto(), recall_tol=0.5)
    expected_delta = abs((2 / 3) - 1.0)

    assert result["verdict"] == "METRIC_REPRODUCTION_WITH_NUMERICAL_VARIATION"
    field = "recall_delta_at_frozen_validation_selected_threshold"
    assert result[field] == expected_delta
    assert json.loads(dump(result))[field] == expected_delta


def test_reproduction_uses_frozen_validation_threshold_not_internal_test_selection():
    committed = [
        _pred("p", "CACHE_EQUIVALENT", 0.8),
        _pred("n", "NOT_CACHE_EQUIVALENT", 0.7),
    ]
    candidate = [
        _pred("p", "CACHE_EQUIVALENT", 0.6),
        _pred("n", "NOT_CACHE_EQUIVALENT", 0.7),
    ]

    result = reproduction_classify(
        candidate,
        committed,
        protocol=_proto(tau=0.5),
        score_tol=0.3,
        recall_tol=0.1,
    )

    assert result["verdict"] == "METRIC_REPRODUCTION_WITH_NUMERICAL_VARIATION"
    assert result["protocol_version"] == 2
    assert result["experiment_id"] == "xp-test"
    assert result["operating_point"] == "fpr_le_0.01"
    assert result["selection_split"] == "validation"
    assert result["target_fpr"] == 0.01
    assert result["frozen_validation_selected_threshold"] == 0.5
    assert result["recall_delta_at_frozen_validation_selected_threshold"] == 0.0
    assert "recall_delta_at_val_fpr1pct" not in result


def test_reproduction_recall_uses_committed_labels_not_candidate_labels():
    committed = [
        _pred("p", "CACHE_EQUIVALENT", 0.9),
        _pred("n", "NOT_CACHE_EQUIVALENT", 0.1),
    ]
    candidate = [
        _pred("p", "NOT_CACHE_EQUIVALENT", 0.8),
        _pred("n", "CACHE_EQUIVALENT", 0.1),
    ]

    result = reproduction_classify(
        candidate,
        committed,
        protocol=_proto(tau=0.5),
        score_tol=0.2,
        recall_tol=0.1,
    )

    assert result["recall_delta_at_frozen_validation_selected_threshold"] == 0.0
    assert result["verdict"] == "METRIC_REPRODUCTION_WITH_NUMERICAL_VARIATION"


@pytest.mark.parametrize("duplicate_side", ["candidate", "committed"])
def test_reproduction_classify_rejects_duplicate_pair_ids(duplicate_side: str):
    unique = [_pred("a", "CACHE_EQUIVALENT", 0.9), _pred("b", "NOT_CACHE_EQUIVALENT", 0.1)]
    duplicated = [_pred("a", "CACHE_EQUIVALENT", 0.9), _pred("a", "CACHE_EQUIVALENT", 0.9)]

    candidate = duplicated if duplicate_side == "candidate" else unique
    committed = duplicated if duplicate_side == "committed" else unique
    result = reproduction_classify(candidate, committed, protocol=_proto())

    assert result == {
        "verdict": "MATERIAL_REPRODUCTION_DRIFT",
        "reason": f"duplicate pair_id in {duplicate_side} predictions: a",
    }


def test_reproduction_classify_rejects_missing_and_unexpected_pair_ids_before_comparison():
    committed = [_pred("a", "CACHE_EQUIVALENT", 0.9), _pred("b", "NOT_CACHE_EQUIVALENT", 0.1)]
    candidate = [_pred("a", "CACHE_EQUIVALENT", 0.9), _pred("c", "NOT_CACHE_EQUIVALENT", 0.1)]

    result = reproduction_classify(candidate, committed, protocol=_proto())

    assert result == {
        "verdict": "MATERIAL_REPRODUCTION_DRIFT",
        "reason": "pair-id set differs; missing_from_candidate=b; unexpected_in_candidate=c",
    }


@pytest.mark.parametrize(
    ("candidate", "committed", "message"),
    [
        ([], [_pred("a", "NOT_CACHE_EQUIVALENT", 0.0)], "candidate predictions are empty"),
        ([_pred("a", "NOT_CACHE_EQUIVALENT", 0.0)], [], "committed predictions are empty"),
        ([], [], "candidate predictions are empty"),
    ],
)
def test_reproduction_classify_rejects_empty_inputs(
    candidate: list[BaselinePrediction], committed: list[BaselinePrediction], message: str
):
    from polarity import transfer as transfer_mod

    with pytest.raises(transfer_mod.ReproductionError, match=message):
        reproduction_classify(candidate, committed, protocol=_proto())


def test_committed_reproduction_report_uses_bounded_score_agreement_verdict():
    run_id = "xp-006-deberta-transformers-fixed"
    committed = load_preds(Path(f"predictions/m3/{run_id}/internal_test.jsonl"))
    protocol = FrozenProtocol.model_validate_json(
        Path(f"reports/m6/protocol-{run_id}.json").read_text(encoding="utf-8")
    )
    report_path = Path(f"reports/m6/reproduction-{run_id}.json")
    report_text = report_path.read_text(encoding="utf-8")
    regenerated = reproduction_classify(committed, committed, protocol=protocol)

    assert dump(regenerated) == report_text
    assert regenerated == json.loads(report_text)
    assert regenerated["verdict"] == "SCORE_AGREEMENT_WITHIN_TOLERANCE"
    assert regenerated["score_agreement_rule"] == (
        "aligned per-row absolute score difference < 1e-6"
    )
    assert regenerated["evidence_scope"] == (
        "score agreement only; does not establish checkpoint identity or retraining reproducibility"
    )
    assert "EXACT_" + "REPRODUCTION" not in report_text


def test_transfer_reproduce_cli_emits_bounded_score_agreement_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    predictions = [_pred("a", "CACHE_EQUIVALENT", 0.9), _pred("b", "NOT_CACHE_EQUIVALENT", 0.1)]
    candidate_path = tmp_path / "candidate.jsonl"
    candidate_path.write_text(
        "\n".join(pred.model_dump_json() for pred in predictions) + "\n", encoding="utf-8"
    )
    committed_path = tmp_path / "predictions/m3/xp-cli/internal_test.jsonl"
    committed_path.parent.mkdir(parents=True)
    committed_path.write_text(candidate_path.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr("polarity.cli.M6_PROTOCOL", _write_proto(tmp_path))
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        app, ["transfer", "reproduce", str(candidate_path), "--committed", "xp-cli"]
    )

    assert result.exit_code == 0
    assert "SCORE_AGREEMENT_WITHIN_TOLERANCE" in result.output
    assert "EXACT_" + "REPRODUCTION" not in result.output
    written = (tmp_path / "reports/m6/reproduction-xp-cli.json").read_text(encoding="utf-8")
    assert "SCORE_AGREEMENT_WITHIN_TOLERANCE" in written


def test_transfer_reproduce_cli_rejects_empty_candidate_without_writing_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    candidate_path = tmp_path / "empty.jsonl"
    candidate_path.write_text("", encoding="utf-8")
    committed_path = tmp_path / "predictions/m3/xp-cli/internal_test.jsonl"
    committed_path.parent.mkdir(parents=True)
    committed_path.write_text(
        _pred("a", "NOT_CACHE_EQUIVALENT", 0.0).model_dump_json() + "\n", encoding="utf-8"
    )
    monkeypatch.setattr("polarity.cli.M6_PROTOCOL", _write_proto(tmp_path))
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        app, ["transfer", "reproduce", str(candidate_path), "--committed", "xp-cli"]
    )

    assert result.exit_code == 1
    assert result.output == "REPRODUCTION FAILED: candidate predictions are empty\n"
    assert not (tmp_path / "reports/m6/reproduction-xp-cli.json").exists()


def test_transfer_reproduce_cli_rejects_protocol_for_different_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    row = _pred("a", "NOT_CACHE_EQUIVALENT", 0.1).model_dump_json() + "\n"
    candidate_path = tmp_path / "candidate.jsonl"
    candidate_path.write_text(row, encoding="utf-8")
    committed_path = tmp_path / "predictions/m3/xp-cli/internal_test.jsonl"
    committed_path.parent.mkdir(parents=True)
    committed_path.write_text(row, encoding="utf-8")
    wrong_protocol = tmp_path / "protocol.json"
    wrong_protocol.write_text(
        dump(_proto(experiment_id="xp-other").model_dump(mode="json")), encoding="utf-8"
    )
    monkeypatch.setattr("polarity.cli.M6_PROTOCOL", wrong_protocol)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        app, ["transfer", "reproduce", str(candidate_path), "--committed", "xp-cli"]
    )

    assert result.exit_code == 1
    assert result.output == (
        "REPRODUCTION FAILED: frozen protocol experiment_id xp-other does not match "
        "committed run xp-cli\n"
    )
    assert not (tmp_path / "reports/m6/reproduction-xp-cli.json").exists()


@pytest.mark.parametrize(
    ("missing_kind", "expected"),
    [
        ("candidate", "cannot read candidate predictions"),
        ("committed", "cannot read committed predictions"),
        ("protocol", "cannot read frozen protocol"),
    ],
)
def test_transfer_reproduce_cli_normalizes_missing_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_kind: str,
    expected: str,
):
    candidate_path = tmp_path / "candidate.jsonl"
    committed_path = tmp_path / "predictions/m3/xp-cli/internal_test.jsonl"
    protocol_path = _write_proto(tmp_path)
    valid_row = _pred("a", "NOT_CACHE_EQUIVALENT", 0.1).model_dump_json() + "\n"
    if missing_kind != "candidate":
        candidate_path.write_text(valid_row, encoding="utf-8")
    if missing_kind != "committed":
        committed_path.parent.mkdir(parents=True)
        committed_path.write_text(valid_row, encoding="utf-8")
    if missing_kind == "protocol":
        protocol_path.unlink()
    monkeypatch.setattr("polarity.cli.M6_PROTOCOL", protocol_path)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        app, ["transfer", "reproduce", str(candidate_path), "--committed", "xp-cli"]
    )

    assert result.exit_code == 1
    assert result.output.startswith(f"REPRODUCTION FAILED: {expected}")
    assert "Traceback" not in result.output
    assert not (tmp_path / "reports/m6/reproduction-xp-cli.json").exists()


@pytest.mark.parametrize("malformed_kind", ["candidate", "committed", "protocol"])
def test_transfer_reproduce_cli_normalizes_malformed_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, malformed_kind: str
):
    candidate_path = tmp_path / "candidate.jsonl"
    committed_path = tmp_path / "predictions/m3/xp-cli/internal_test.jsonl"
    committed_path.parent.mkdir(parents=True)
    valid_row = _pred("a", "NOT_CACHE_EQUIVALENT", 0.1).model_dump_json() + "\n"
    candidate_path.write_text(valid_row, encoding="utf-8")
    committed_path.write_text(valid_row, encoding="utf-8")
    protocol_path = _write_proto(tmp_path)
    if malformed_kind == "protocol":
        protocol_path.write_text("{not-json\n", encoding="utf-8")
    else:
        path = candidate_path if malformed_kind == "candidate" else committed_path
        path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr("polarity.cli.M6_PROTOCOL", protocol_path)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        app, ["transfer", "reproduce", str(candidate_path), "--committed", "xp-cli"]
    )

    assert result.exit_code == 1
    assert "REPRODUCTION FAILED:" in result.output
    assert "invalid" in result.output
    assert "Traceback" not in result.output
    assert not (tmp_path / "reports/m6/reproduction-xp-cli.json").exists()


def test_transfer_reproduce_cli_normalizes_overflowing_calibrator_parameter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    candidate_path = tmp_path / "candidate.jsonl"
    committed_path = tmp_path / "predictions/m3/xp-cli/internal_test.jsonl"
    committed_path.parent.mkdir(parents=True)
    valid_row = _pred("a", "NOT_CACHE_EQUIVALENT", 0.1).model_dump_json() + "\n"
    candidate_path.write_text(valid_row, encoding="utf-8")
    committed_path.write_text(valid_row, encoding="utf-8")
    protocol = _proto(experiment_id="xp-cli").model_dump(mode="json")
    protocol["calibrator_method"] = "temperature"
    protocol["calibrator_params"] = {"temperature": 10**400}
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    monkeypatch.setattr("polarity.cli.M6_PROTOCOL", protocol_path)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        app, ["transfer", "reproduce", str(candidate_path), "--committed", "xp-cli"]
    )

    assert result.exit_code == 1
    assert result.output.startswith("REPRODUCTION FAILED: invalid frozen protocol")
    assert "Traceback" not in result.output
    assert not (tmp_path / "reports/m6/reproduction-xp-cli.json").exists()


@pytest.mark.parametrize("invalid_kind", ["candidate", "committed", "protocol"])
def test_transfer_reproduce_cli_normalizes_invalid_utf8_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid_kind: str
):
    candidate_path = tmp_path / "candidate.jsonl"
    committed_path = tmp_path / "predictions/m3/xp-cli/internal_test.jsonl"
    committed_path.parent.mkdir(parents=True)
    valid_row = _pred("a", "NOT_CACHE_EQUIVALENT", 0.1).model_dump_json() + "\n"
    candidate_path.write_text(valid_row, encoding="utf-8")
    committed_path.write_text(valid_row, encoding="utf-8")
    protocol_path = _write_proto(tmp_path)
    invalid_path = {
        "candidate": candidate_path,
        "committed": committed_path,
        "protocol": protocol_path,
    }[invalid_kind]
    invalid_path.write_bytes(b"\xff")
    monkeypatch.setattr("polarity.cli.M6_PROTOCOL", protocol_path)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        app, ["transfer", "reproduce", str(candidate_path), "--committed", "xp-cli"]
    )

    assert result.exit_code == 1
    assert result.output.startswith("REPRODUCTION FAILED:")
    assert "Traceback" not in result.output
    assert not (tmp_path / "reports/m6/reproduction-xp-cli.json").exists()


def test_different_answer_probe_is_valid_and_balanced():
    # The M6.1 diagnostic probe: schema-valid golden pairs, documented composition.
    from pathlib import Path

    from polarity.data.loaders import load_golden

    probe = load_golden(Path("data/diagnostics/different-answer-probe.json"))
    assert len(probe) == 44
    mc = [p for p in probe if not p.equivalent]
    controls = [p for p in probe if p.equivalent]
    assert len(mc) == 36 and len(controls) == 8
    # six meaning-change sub-types, 6 each
    from collections import Counter

    counts = Counter(p.pairType for p in mc)
    assert len(counts) == 6 and all(v == 6 for v in counts.values())
    assert all(p.pairType == "paraphrase-should-hit" for p in controls)
