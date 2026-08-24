"""M6.1 report/provenance gates: public MRPC scores stay label- and text-free."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

import polarity.m61_report as m61_mod
from polarity.cli import app
from polarity.m61_report import (
    M61ReportError,
    MrpcScore,
    MrpcSourceContract,
    build_evaluation_binding,
    build_mrpc_provenance,
    build_mrpc_report,
    build_probe_report,
    check_evaluation_binding,
    check_mrpc_artifacts,
    check_probe_artifacts,
    load_mrpc_scores,
    verify_mrpc_source,
)
from polarity.transfer import FrozenProtocol

_MRPC_GATE_ARTIFACTS = (
    "reports/m6/model-manifest-xp-006-deberta-transformers-fixed.json",
    "reports/m6/protocol-xp-006-deberta-transformers-fixed.json",
    "predictions/m6.1/mrpc-transfer.jsonl",
    "reports/m6.1/mrpc-transfer.json",
    "reports/m6.1/mrpc-provenance.json",
    "reports/m6.1/mrpc-binding.json",
)


def _source_rows() -> list[dict[str, object]]:
    return [
        {
            "id": "mrpc-test-0",
            "group": "mrpc",
            "base": "Alpha one.",
            "probe": "Alpha one!",
            "pairType": "paraphrase",
            "equivalent": True,
            "note": "",
        },
        {
            "id": "mrpc-test-1",
            "group": "mrpc",
            "base": "Beta",
            "probe": "Gamma",
            "pairType": "not-paraphrase",
            "equivalent": False,
            "note": "",
        },
        {
            "id": "mrpc-test-2",
            "group": "mrpc",
            "base": "Delta",
            "probe": "Delta again",
            "pairType": "paraphrase",
            "equivalent": True,
            "note": "",
        },
        {
            "id": "mrpc-test-3",
            "group": "mrpc",
            "base": "Alpha one",
            "probe": "Alpha one",
            "pairType": "paraphrase",
            "equivalent": True,
            "note": "",
        },
    ]


def _write_source(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "mrpc.json"
    path.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    return path


def _scores() -> list[MrpcScore]:
    return [
        MrpcScore(row_index=0, id="mrpc-test-0", raw_score=0.9, calibrated=0.8),
        MrpcScore(row_index=1, id="mrpc-test-1", raw_score=0.7, calibrated=0.6),
        MrpcScore(row_index=2, id="mrpc-test-2", raw_score=0.8, calibrated=0.7),
        MrpcScore(row_index=3, id="mrpc-test-3", raw_score=0.1, calibrated=0.2),
    ]


def _contract(path: Path, **overrides: object) -> MrpcSourceContract:
    payload: dict[str, object] = {
        "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "row_count": 4,
        "positive_count": 3,
        "negative_count": 1,
        "label_mapping": {"paraphrase": True, "not-paraphrase": False},
        "filtering": "none_all_rows_retained",
        "duplicate_policy": "retain_all_source_rows",
        "normalized_ordered_pair_duplicate_rows": 1,
        "duplicate_label_conflicts": 0,
    }
    payload.update(overrides)
    return MrpcSourceContract.model_validate(payload)


def _protocol(tau: float = 0.75) -> FrozenProtocol:
    observed = {"tp": 1, "fp": 0, "tn": 1, "fn": 0, "recall": 1.0, "fpr": 0.0}
    return FrozenProtocol.model_validate(
        {
            "protocol_version": 2,
            "experiment_id": "xp-test",
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


def _copy_mrpc_gate_artifacts(root: Path) -> None:
    for relative in _MRPC_GATE_ARTIFACTS:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(relative, destination)


def _rewrite_mrpc_companion_hashes(root: Path) -> None:
    prediction_path = root / "predictions/m6.1/mrpc-transfer.jsonl"
    report_path = root / "reports/m6.1/mrpc-transfer.json"
    provenance_path = root / "reports/m6.1/mrpc-provenance.json"
    binding_path = root / "reports/m6.1/mrpc-binding.json"
    protocol_path = root / "reports/m6/protocol-xp-006-deberta-transformers-fixed.json"
    manifest_path = root / "reports/m6/model-manifest-xp-006-deberta-transformers-fixed.json"

    provenance = build_mrpc_provenance(
        prediction_path=prediction_path,
        report_path=report_path,
        protocol_path=protocol_path,
        model_manifest_path=manifest_path,
    )
    provenance_path.write_text(
        json.dumps(provenance, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    binding = build_evaluation_binding(
        evaluation_id="m6.1-mrpc-test",
        source_identifier=m61_mod.MRPC_SOURCE_IDENTIFIER,
        source_sha256=m61_mod.MRPC_SOURCE_CONTRACT.raw_sha256,
        source_rows=m61_mod.MRPC_SOURCE_CONTRACT.row_count,
        prediction_path=prediction_path,
        prediction_rows=m61_mod.MRPC_SOURCE_CONTRACT.row_count,
        report_path=report_path,
        protocol_path=protocol_path,
        model_manifest_path=manifest_path,
    )
    binding["source_provenance"] = {
        "path": provenance_path.as_posix(),
        "sha256": hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
    }
    binding_path.write_text(json.dumps(binding, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _mutate_mrpc_report(root: Path, mutation: str) -> None:
    report_path = root / "reports/m6.1/mrpc-transfer.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if mutation == "missing_threshold_free":
        del report["threshold_free"]
    elif mutation == "extra_threshold_free_field":
        report["threshold_free"]["unreviewed_metric"] = 0.5
    elif mutation == "wrong_evaluation_id":
        report["evaluation_id"] = "different-evaluation"
    elif mutation == "wrong_protocol_version":
        report["protocol_version"] = 1
    elif mutation == "wrong_experiment_id":
        report["experiment_id"] = "xp-other"
    elif mutation == "wrong_retention":
        report["row_retention"] = "filtered"
    elif mutation == "wrong_duplicate_policy":
        report["duplicate_policy"]["policy"] = "drop_duplicates"
    elif mutation == "missing_operating_point":
        del report["at_frozen_m6_operating_points"]["zero_fp"]
    elif mutation == "wrong_operating_threshold":
        report["at_frozen_m6_operating_points"]["fpr_le_0.01"]["tau"] = 0.5
    elif mutation == "inconsistent_confusion_count":
        report["at_frozen_m6_operating_points"]["fpr_le_0.01"]["tp"] += 1
    elif mutation == "inconsistent_rate":
        report["at_frozen_m6_operating_points"]["fpr_le_0.01"]["recall"] = 0.5
    elif mutation == "missing_distribution_shape":
        del report["raw_score_distributions"]["paraphrase"]["q05"]
    elif mutation == "out_of_range_distribution":
        report["raw_score_distributions"]["not_paraphrase"]["mean"] = 1.1
    elif mutation == "missing_recall_budget":
        del report["threshold_free"]["recall_at_fpr_budgets"]["0.02"]
    elif mutation == "wrong_disposition":
        report["disposition"] = "UNREVIEWED"
    elif mutation == "string_top_level_count":
        report["n_pairs"] = str(report["n_pairs"])
    elif mutation == "string_confusion_count":
        point = report["at_frozen_m6_operating_points"]["fpr_le_0.01"]
        point["tp"] = str(point["tp"])
    elif mutation == "string_rate":
        point = report["at_frozen_m6_operating_points"]["fpr_le_0.01"]
        point["recall"] = str(point["recall"])
    elif mutation == "string_distribution_count":
        distribution = report["raw_score_distributions"]["paraphrase"]
        distribution["n"] = str(distribution["n"])
    elif mutation == "boolean_duplicate_count":
        report["duplicate_policy"]["label_conflicts"] = False
    else:  # pragma: no cover - the parameter list below is the closed mutation inventory
        raise AssertionError(mutation)
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    _rewrite_mrpc_companion_hashes(root)


def test_mrpc_score_schema_keeps_only_stable_order_identity_and_scores(tmp_path: Path):
    path = tmp_path / "scores.jsonl"
    path.write_text(
        '{"row_index":0,"id":"mrpc-test-0","raw_score":0.75,"calibrated":0.5}\n',
        encoding="utf-8",
    )

    scores = load_mrpc_scores(path)

    assert scores == [MrpcScore(row_index=0, id="mrpc-test-0", raw_score=0.75, calibrated=0.5)]


@pytest.mark.parametrize("forbidden", ["gold_equivalent", "pair_type", "group", "base", "probe"])
def test_mrpc_score_schema_rejects_private_or_gold_fields(forbidden: str):
    payload = {
        "row_index": 0,
        "id": "mrpc-test-0",
        "raw_score": 0.75,
        "calibrated": 0.5,
        forbidden: False,
    }

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        MrpcScore.model_validate(payload)


def test_mrpc_score_identity_must_match_row_index():
    with pytest.raises(ValidationError, match="id must equal mrpc-test-7"):
        MrpcScore(row_index=7, id="mrpc-test-8", raw_score=0.75, calibrated=0.5)


def test_load_mrpc_scores_normalizes_schema_errors_for_release_gate_output(tmp_path: Path):
    path = tmp_path / "scores.jsonl"
    path.write_text(
        '{"row_index":0,"id":"mrpc-test-0","raw_score":0.75,"calibrated":0.5,'
        '"gold_equivalent":true}\n',
        encoding="utf-8",
    )

    with pytest.raises(M61ReportError, match="invalid MRPC score row 1"):
        load_mrpc_scores(path)


def test_load_mrpc_scores_normalizes_invalid_utf8_for_release_gate_output(tmp_path: Path):
    path = tmp_path / "scores.jsonl"
    path.write_bytes(b"\xff")

    with pytest.raises(M61ReportError, match="cannot read MRPC scores"):
        load_mrpc_scores(path)


def test_verify_mrpc_source_joins_checksum_verified_labels_to_score_order(tmp_path: Path):
    source = _write_source(tmp_path, _source_rows())

    rows = verify_mrpc_source(source, _scores(), _contract(source))

    assert [(row.id, row.gold_equivalent, row.raw_score) for row in rows] == [
        ("mrpc-test-0", True, 0.9),
        ("mrpc-test-1", False, 0.7),
        ("mrpc-test-2", True, 0.8),
        ("mrpc-test-3", True, 0.1),
    ]


def test_verify_mrpc_source_rejects_checksum_mismatch(tmp_path: Path):
    source = _write_source(tmp_path, _source_rows())
    contract = _contract(source, raw_sha256="0" * 64)

    with pytest.raises(M61ReportError, match="MRPC source checksum mismatch"):
        verify_mrpc_source(source, _scores(), contract)


def test_verify_mrpc_source_normalizes_invalid_utf8(tmp_path: Path):
    source = tmp_path / "mrpc.json"
    source.write_bytes(b"\xff")

    with pytest.raises(M61ReportError, match="invalid MRPC source: cannot read"):
        verify_mrpc_source(source, _scores(), _contract(source))


def test_verify_mrpc_source_rejects_source_order_mismatch(tmp_path: Path):
    rows = _source_rows()
    rows[0], rows[1] = rows[1], rows[0]
    source = _write_source(tmp_path, rows)

    with pytest.raises(M61ReportError, match="source row 0 must have id mrpc-test-0"):
        verify_mrpc_source(source, _scores(), _contract(source))


def test_verify_mrpc_source_rejects_label_mapping_error(tmp_path: Path):
    rows = _source_rows()
    rows[1]["equivalent"] = True
    source = _write_source(tmp_path, rows)

    with pytest.raises(M61ReportError, match="label mapping mismatch for mrpc-test-1"):
        verify_mrpc_source(source, _scores(), _contract(source))


def test_verify_mrpc_source_rejects_missing_prediction_id(tmp_path: Path):
    source = _write_source(tmp_path, _source_rows())

    with pytest.raises(M61ReportError, match=r"prediction ID set differs.*missing mrpc-test-3"):
        verify_mrpc_source(source, _scores()[:-1], _contract(source))


def test_verify_mrpc_source_rejects_duplicate_source_id(tmp_path: Path):
    rows = _source_rows()
    rows[2]["id"] = "mrpc-test-1"
    source = _write_source(tmp_path, rows)

    with pytest.raises(M61ReportError, match="duplicate source id: mrpc-test-1"):
        verify_mrpc_source(source, _scores(), _contract(source))


def test_verify_mrpc_source_enforces_retained_duplicate_policy(tmp_path: Path):
    source = _write_source(tmp_path, _source_rows())
    contract = _contract(source, normalized_ordered_pair_duplicate_rows=0)

    with pytest.raises(M61ReportError, match="normalized ordered-pair duplicate count"):
        verify_mrpc_source(source, _scores(), contract)


def test_build_mrpc_report_reproduces_ranking_and_frozen_threshold_metrics(tmp_path: Path):
    source = _write_source(tmp_path, _source_rows())
    contract = _contract(source)
    verified = verify_mrpc_source(source, _scores(), contract)

    report = build_mrpc_report(verified, _protocol(), contract)

    assert report["schema_version"] == "polarity-m61-mrpc-report-1"
    assert report["metric_basis"] == "checksum_verified_official_local_source"
    assert report["n_pairs"] == 4
    assert report["n_paraphrase"] == 3
    assert report["n_not_paraphrase"] == 1
    assert report["threshold_free"]["roc_auc"] == 0.6666667
    assert report["threshold_free"]["custom_step_pauc_fpr_0.02"] == 0.666667
    primary = report["at_frozen_m6_operating_points"]["fpr_le_0.01"]
    assert primary == {
        "tau": 0.75,
        "tp": 2,
        "fp": 0,
        "tn": 1,
        "fn": 1,
        "recall": 0.666667,
        "fpr": 0.0,
        "precision": 1.0,
    }
    assert report["artifact_identity"] == {
        "checkpoint_sha256": "a" * 64,
        "artifact_bundle_fingerprint": f"pl1:{'b' * 64}",
        "recipe_fingerprint": f"pl1:{'c' * 64}",
        "execution_fingerprint": f"pl1:{'d' * 64}",
    }


def test_build_mrpc_report_rejects_empty_verified_rows(tmp_path: Path):
    source = _write_source(tmp_path, _source_rows())

    with pytest.raises(M61ReportError, match="no verified MRPC rows"):
        build_mrpc_report([], _protocol(), _contract(source))


def test_build_probe_report_reproduces_group_acceptance_counts(tmp_path: Path):
    source_rows = [
        {
            "id": "entity-1",
            "group": "entity",
            "base": "",
            "probe": "",
            "pairType": "entity-swap",
            "equivalent": False,
            "note": "",
        },
        {
            "id": "measurement-1",
            "group": "measurement",
            "base": "",
            "probe": "",
            "pairType": "measurement-change",
            "equivalent": False,
            "note": "",
        },
        {
            "id": "control-1",
            "group": "control-para",
            "base": "",
            "probe": "",
            "pairType": "paraphrase-should-hit",
            "equivalent": True,
            "note": "",
        },
    ]
    source = _write_source(tmp_path, source_rows)
    scores = [
        m61_mod.ProbeScore(
            row_index=0,
            id="entity-1",
            raw_score=0.8,
            calibrated=0.7,
        ),
        m61_mod.ProbeScore(
            row_index=1,
            id="measurement-1",
            raw_score=0.9,
            calibrated=0.85,
        ),
        m61_mod.ProbeScore(
            row_index=2,
            id="control-1",
            raw_score=0.95,
            calibrated=0.9,
        ),
    ]

    report = build_probe_report(source, scores, _protocol(tau=0.85))

    assert report["schema_version"] == "polarity-m61-probe-report-1"
    assert report["diagnostic_scope"] == "author_created_post_hoc_not_a_benchmark"
    primary = report["operating_points"]["fpr_le_0.01"]
    assert primary["by_group"]["entity"]["false_accepts"] == 0
    assert primary["by_group"]["measurement"]["false_accepts"] == 1
    assert primary["by_group"]["control-para"]["recalled"] == 1


@pytest.mark.parametrize("forbidden", ["group", "pair_type", "gold_equivalent", "base", "probe"])
def test_probe_score_schema_rejects_non_score_fields(forbidden: str):
    payload = {
        "row_index": 0,
        "id": "entity-1",
        "raw_score": 0.8,
        "calibrated": 0.7,
        forbidden: False,
    }

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        m61_mod.ProbeScore.model_validate(payload)


def test_load_probe_scores_rejects_non_sequential_row_index(tmp_path: Path):
    path = tmp_path / "probe-scores.jsonl"
    path.write_text(
        '{"row_index":1,"id":"entity-1","raw_score":0.8,"calibrated":0.7}\n',
        encoding="utf-8",
    )

    with pytest.raises(M61ReportError, match="row_index 1 must equal line order 0"):
        m61_mod.load_probe_scores(path)


def test_build_probe_report_rejects_prediction_order_drift(tmp_path: Path):
    source = _write_source(
        tmp_path,
        [
            {
                "id": "entity-1",
                "group": "entity",
                "base": "",
                "probe": "",
                "pairType": "entity-swap",
                "equivalent": False,
                "note": "",
            },
            {
                "id": "control-1",
                "group": "control-para",
                "base": "",
                "probe": "",
                "pairType": "paraphrase-should-hit",
                "equivalent": True,
                "note": "",
            },
        ],
    )
    scores = [
        m61_mod.ProbeScore(
            row_index=0,
            id="control-1",
            raw_score=0.95,
            calibrated=0.9,
        ),
        m61_mod.ProbeScore(
            row_index=1,
            id="entity-1",
            raw_score=0.8,
            calibrated=0.7,
        ),
    ]

    with pytest.raises(M61ReportError, match="probe prediction order differs from source"):
        build_probe_report(source, scores, _protocol())


@pytest.mark.parametrize(
    ("ids", "message"),
    [
        (["entity-1", "entity-1"], "duplicate probe prediction id: entity-1"),
        (["entity-1"], "probe prediction ID set differs: missing control-1"),
        (["entity-1", "control-1", "extra-1"], "probe prediction ID set differs: extra extra-1"),
    ],
)
def test_build_probe_report_rejects_prediction_row_set_tamper(
    tmp_path: Path, ids: list[str], message: str
):
    source = _write_source(
        tmp_path,
        [
            {
                "id": "entity-1",
                "group": "entity",
                "base": "",
                "probe": "",
                "pairType": "entity-swap",
                "equivalent": False,
                "note": "",
            },
            {
                "id": "control-1",
                "group": "control-para",
                "base": "",
                "probe": "",
                "pairType": "paraphrase-should-hit",
                "equivalent": True,
                "note": "",
            },
        ],
    )
    scores = [
        m61_mod.ProbeScore(
            row_index=index,
            id=score_id,
            raw_score=0.8,
            calibrated=0.7,
        )
        for index, score_id in enumerate(ids)
    ]

    with pytest.raises(M61ReportError, match=message):
        build_probe_report(source, scores, _protocol())


@pytest.mark.parametrize("coercible_label", [0, "false"])
def test_build_probe_report_rejects_coercible_source_labels(
    tmp_path: Path, coercible_label: object
):
    source = _write_source(
        tmp_path,
        [
            {
                "id": "entity-1",
                "group": "entity",
                "base": "",
                "probe": "",
                "pairType": "entity-swap",
                "equivalent": coercible_label,
                "note": "",
            }
        ],
    )
    scores = [
        m61_mod.ProbeScore(
            row_index=0,
            id="entity-1",
            raw_score=0.8,
            calibrated=0.7,
        )
    ]

    with pytest.raises(M61ReportError, match=r"invalid M6\.1 probe source row"):
        build_probe_report(source, scores, _protocol())


def test_build_evaluation_binding_uses_protocol_and_model_manifest_identity(tmp_path: Path):
    source = tmp_path / "source.json"
    prediction = tmp_path / "prediction.jsonl"
    report = tmp_path / "report.json"
    source.write_text("[]\n", encoding="utf-8")
    prediction.write_text("{}\n", encoding="utf-8")
    report.write_text("{}\n", encoding="utf-8")

    binding = build_evaluation_binding(
        evaluation_id="test-eval",
        source_identifier="fixture-source",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_rows=0,
        prediction_path=prediction,
        prediction_rows=1,
        report_path=report,
        protocol_path=Path("reports/m6/protocol-xp-006-deberta-transformers-fixed.json"),
        model_manifest_path=Path(
            "reports/m6/model-manifest-xp-006-deberta-transformers-fixed.json"
        ),
    )

    assert binding["artifact_identity"]["checkpoint_sha256"] == (
        "8d765a7e55c94bb5a8c7754eb705ddf8dc7e6c74cc72e1b0ce4c9110e7a0f355"
    )
    assert binding["source"] == {
        "identifier": "fixture-source",
        "sha256": hashlib.sha256(b"[]\n").hexdigest(),
        "rows": 0,
    }
    assert binding["predictions"]["sha256"] == hashlib.sha256(b"{}\n").hexdigest()
    assert binding["protocol"]["protocol_version"] == 2


def test_check_evaluation_binding_detects_prediction_hash_drift(tmp_path: Path):
    source = tmp_path / "source.json"
    prediction = tmp_path / "prediction.jsonl"
    report = tmp_path / "report.json"
    binding_path = tmp_path / "binding.json"
    source.write_text("[]\n", encoding="utf-8")
    prediction.write_text("{}\n", encoding="utf-8")
    report.write_text("{}\n", encoding="utf-8")
    kwargs = {
        "evaluation_id": "test-eval",
        "source_identifier": "fixture-source",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_rows": 0,
        "prediction_path": prediction,
        "prediction_rows": 1,
        "report_path": report,
        "protocol_path": Path("reports/m6/protocol-xp-006-deberta-transformers-fixed.json"),
        "model_manifest_path": Path(
            "reports/m6/model-manifest-xp-006-deberta-transformers-fixed.json"
        ),
    }
    binding_path.write_text(
        json.dumps(build_evaluation_binding(**kwargs), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    prediction.write_text('{"changed":true}\n', encoding="utf-8")

    with pytest.raises(M61ReportError, match="evaluation binding drift"):
        check_evaluation_binding(binding_path, **kwargs)


def test_build_mrpc_provenance_records_exact_reconstructed_source_and_release_hashes():
    provenance = build_mrpc_provenance(
        prediction_path=Path("predictions/m6.1/mrpc-transfer.jsonl"),
        report_path=Path("reports/m6.1/mrpc-transfer.json"),
        protocol_path=Path("reports/m6/protocol-xp-006-deberta-transformers-fixed.json"),
        model_manifest_path=Path(
            "reports/m6/model-manifest-xp-006-deberta-transformers-fixed.json"
        ),
    )

    source = provenance["source"]
    assert (
        source["official_url"] == "https://www.microsoft.com/en-us/download/details.aspx?id=52398"
    )
    assert source["loader"] == "nyu-mll/glue"
    assert source["config"] == "mrpc"
    assert source["split"] == "test"
    assert source["revision"] == "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
    assert source["raw_sha256"] == (
        "7999ee0ba2af1c43e416c35d152bdcaa99acb79ae1791914e748535f78d8a18c"
    )
    assert source["cached_arrow_sha256"] == (
        "0152c178bc09c083804600f657ae610005dfce932f29435cd4938509f9643d71"
    )
    assert provenance["distribution"] == {
        "rows": 1725,
        "paraphrase": 1147,
        "not_paraphrase": 578,
    }
    assert provenance["duplicate_policy"]["normalized_ordered_pair_duplicate_rows"] == 1
    assert provenance["artifact_identity"]["execution_fingerprint"].startswith("pl1:")


def test_committed_probe_report_and_veritas_probe_bindings_are_drift_free():
    check_probe_artifacts()
    report = json.loads(
        Path("reports/m6.1/different-answer-probe.json").read_text(encoding="utf-8")
    )
    primary = report["operating_points"]["fpr_le_0.01"]["by_group"]
    assert {group: primary[group]["false_accepts"] for group in primary} == {
        "comparative": 0,
        "control-para": 0,
        "entity": 0,
        "measurement": 4,
        "numeric": 1,
        "operation": 0,
        "temporal": 2,
    }
    assert primary["control-para"]["recalled"] == 8


def test_probe_gate_reads_source_and_predictions_once(
    monkeypatch: pytest.MonkeyPatch,
):
    source_path = Path("data/diagnostics/different-answer-probe.json").resolve()
    prediction_path = Path("predictions/m6.1/different-answer-probe.jsonl").resolve()
    tracked_paths = {source_path, prediction_path}
    read_counts = {source_path: 0, prediction_path: 0}
    real_read_bytes = Path.read_bytes
    real_read_text = Path.read_text

    def counted_read_bytes(path: Path) -> bytes:
        resolved = path.resolve()
        if resolved in tracked_paths:
            read_counts[resolved] += 1
            if read_counts[resolved] > 1:
                raise AssertionError(f"second read of {resolved}")
        return real_read_bytes(path)

    def counted_read_text(
        path: Path, encoding: str | None = None, errors: str | None = None
    ) -> str:
        resolved = path.resolve()
        if resolved in tracked_paths:
            read_counts[resolved] += 1
            if read_counts[resolved] > 1:
                raise AssertionError(f"second read of {resolved}")
        return real_read_text(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
    monkeypatch.setattr(Path, "read_text", counted_read_text)

    check_probe_artifacts()

    assert read_counts == {source_path: 1, prediction_path: 1}


def test_tracked_m61_prediction_directory_has_no_gold_or_pair_type_fields():
    result = subprocess.run(
        [
            "git",
            "grep",
            "-n",
            "-E",
            '"gold_equivalent"|"pair_type"',
            "--",
            "predictions/m6.1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1, result.stdout
    assert result.stdout == ""


@pytest.mark.parametrize("failure", ["missing", "invalid_utf8", "malformed_json", "schema_error"])
def test_probe_score_artifact_failures_are_normalized_for_release_gate_output(
    tmp_path: Path, failure: str
):
    for relative in (
        "reports/m6/model-manifest-xp-006-deberta-transformers-fixed.json",
        "reports/m6/protocol-xp-006-deberta-transformers-fixed.json",
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(relative, destination)
    prediction_path = tmp_path / "predictions/m6.1/different-answer-probe.jsonl"
    if failure == "invalid_utf8":
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.write_bytes(b"\xff")
    elif failure == "malformed_json":
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.write_text("{not-json}\n", encoding="utf-8")
    elif failure == "schema_error":
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.write_text(
            '{"id":"probe-0","group":"x","pair_type":"x","gold_equivalent":false,'
            '"raw_score":0.5,"calibrated":0.5,"unexpected":true}\n',
            encoding="utf-8",
        )

    with pytest.raises(M61ReportError, match=r"invalid M6\.1 probe scores"):
        check_probe_artifacts(root=tmp_path)


@pytest.mark.parametrize("artifact_kind", ["probe", "Veritas"])
@pytest.mark.parametrize("failure", ["missing", "invalid_utf8", "malformed"])
def test_probe_report_cli_normalizes_score_artifact_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_kind: str,
    failure: str,
):
    path = tmp_path / f"{artifact_kind.lower()}.jsonl"
    if failure == "invalid_utf8":
        path.write_bytes(b"\xff")
    elif failure == "malformed":
        path.write_text("{}\n", encoding="utf-8")
    constant = "PROBE_PREDICTIONS" if artifact_kind == "probe" else "VERITAS_PREDICTIONS"
    monkeypatch.setattr(m61_mod, constant, path)

    result = CliRunner().invoke(app, ["transfer", "probe-report", "--check"])

    assert result.exit_code == 1
    assert result.output.startswith(f"VERIFY FAILED: invalid M6.1 {artifact_kind} scores:")
    assert "Traceback" not in result.output


@pytest.mark.parametrize("artifact_kind", ["probe", "Veritas"])
@pytest.mark.parametrize("failure", ["missing", "invalid_utf8"])
def test_probe_report_cli_normalizes_source_artifact_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_kind: str,
    failure: str,
):
    constant = "PROBE_SOURCE" if artifact_kind == "probe" else "VERITAS_SOURCE"
    path = tmp_path / f"{failure}-{artifact_kind.lower()}.json"
    if failure == "invalid_utf8":
        path.write_bytes(b"\xff")
    monkeypatch.setattr(m61_mod, constant, path)

    result = CliRunner().invoke(app, ["transfer", "probe-report", "--check"])

    assert result.exit_code == 1
    assert result.output.startswith(f"VERIFY FAILED: invalid M6.1 {artifact_kind} source:")
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("constant", "error_prefix"),
    [
        ("PROBE_REPORT", "cannot read probe report"),
        ("PROBE_BINDING", "cannot read evaluation binding"),
        ("VERITAS_BINDING", "cannot read evaluation binding"),
    ],
)
def test_probe_report_cli_normalizes_invalid_utf8_report_and_binding_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    constant: str,
    error_prefix: str,
):
    path = tmp_path / f"{constant.lower()}.json"
    path.write_bytes(b"\xff")
    monkeypatch.setattr(m61_mod, constant, path)

    result = CliRunner().invoke(app, ["transfer", "probe-report", "--check"])

    assert result.exit_code == 1
    assert result.output.startswith(f"VERIFY FAILED: {error_prefix}")
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("constant", "error_prefix"),
    [
        ("M6_PROTOCOL", "invalid frozen protocol"),
        ("M6_MODEL_MANIFEST", "invalid model manifest identity"),
    ],
)
def test_probe_report_cli_normalizes_invalid_utf8_release_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    constant: str,
    error_prefix: str,
):
    path = tmp_path / f"{constant.lower()}.json"
    path.write_bytes(b"\xff")
    monkeypatch.setattr(m61_mod, constant, path)

    result = CliRunner().invoke(app, ["transfer", "probe-report", "--check"])

    assert result.exit_code == 1
    assert result.output.startswith(f"VERIFY FAILED: {error_prefix}")
    assert "Traceback" not in result.output


def test_committed_mrpc_source_free_gate_is_explicitly_structural_only():
    assert check_mrpc_artifacts() == "structural_only_metrics_not_reproduced"


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_threshold_free",
        "extra_threshold_free_field",
        "wrong_evaluation_id",
        "wrong_protocol_version",
        "wrong_experiment_id",
        "wrong_retention",
        "wrong_duplicate_policy",
        "missing_operating_point",
        "wrong_operating_threshold",
        "inconsistent_confusion_count",
        "inconsistent_rate",
        "missing_distribution_shape",
        "out_of_range_distribution",
        "missing_recall_budget",
        "wrong_disposition",
        "string_top_level_count",
        "string_confusion_count",
        "string_rate",
        "string_distribution_count",
        "boolean_duplicate_count",
    ],
)
def test_mrpc_source_free_gate_rejects_malformed_nested_report_with_valid_hashes(
    tmp_path: Path, mutation: str
):
    _copy_mrpc_gate_artifacts(tmp_path)
    _mutate_mrpc_report(tmp_path, mutation)

    with pytest.raises(M61ReportError, match="invalid MRPC report structure"):
        check_mrpc_artifacts(root=tmp_path)


def test_committed_mrpc_report_records_the_checksum_verified_audit_result():
    report = json.loads(Path("reports/m6.1/mrpc-transfer.json").read_text(encoding="utf-8"))
    assert report["n_pairs"] == 1725
    assert report["n_paraphrase"] == 1147
    assert report["n_not_paraphrase"] == 578
    assert report["threshold_free"]["roc_auc"] == 0.6398699
    assert report["threshold_free"]["custom_step_pauc_fpr_0.02"] == 0.031938
    assert report["at_frozen_m6_operating_points"]["fpr_le_0.01"] == {
        "tau": 0.994833,
        "tp": 898,
        "fp": 332,
        "tn": 246,
        "fn": 249,
        "recall": 0.782912,
        "fpr": 0.574394,
        "precision": 0.730081,
    }
    assert report["raw_score_distributions"]["paraphrase"]["q05"] == 0.042607
    assert report["raw_score_distributions"]["paraphrase"]["q95"] == 0.999687


def test_mrpc_source_free_gate_reports_committed_report_hash_drift(tmp_path: Path):
    for relative in (
        "reports/m6/model-manifest-xp-006-deberta-transformers-fixed.json",
        "reports/m6/protocol-xp-006-deberta-transformers-fixed.json",
        "predictions/m6.1/mrpc-transfer.jsonl",
        "reports/m6.1/mrpc-transfer.json",
        "reports/m6.1/mrpc-provenance.json",
        "reports/m6.1/mrpc-binding.json",
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(relative, destination)
    report_path = tmp_path / "reports/m6.1/mrpc-transfer.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["disposition"] = "CORRUPTED"
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(M61ReportError, match="MRPC report hash drift"):
        check_mrpc_artifacts(root=tmp_path)


@pytest.mark.parametrize(
    ("constant", "error_prefix"),
    [
        ("MRPC_PREDICTIONS", "cannot read MRPC scores"),
        ("MRPC_REPORT", "cannot read MRPC report"),
        ("MRPC_PROVENANCE", "cannot read MRPC provenance"),
        ("MRPC_BINDING", "cannot read MRPC binding"),
    ],
)
def test_mrpc_report_cli_normalizes_invalid_utf8_public_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    constant: str,
    error_prefix: str,
):
    path = tmp_path / f"{constant.lower()}.json"
    path.write_bytes(b"\xff")
    monkeypatch.setattr(m61_mod, constant, path)

    result = CliRunner().invoke(app, ["transfer", "mrpc-report", "--check"])

    assert result.exit_code == 1
    assert result.output.startswith(f"VERIFY FAILED: {error_prefix}")
    assert "Traceback" not in result.output


def test_committed_mrpc_full_gate_reproduces_metrics_when_official_source_is_available():
    source = Path("data/raw/mrpc/mrpc-test-golden.json")
    if not source.exists():
        pytest.skip("author-only ignored MRPC source is unavailable")

    assert check_mrpc_artifacts(source) == "full_metrics_reproduced"


def test_probe_report_cli_check_succeeds_keylessly():
    result = CliRunner().invoke(app, ["transfer", "probe-report", "--check"])

    assert result.exit_code == 0, result.output
    assert "OK: M6.1 probe report and evaluation bindings reproduce byte-for-byte" in result.output


def test_mrpc_report_cli_source_free_check_does_not_claim_metric_reproduction():
    result = CliRunner().invoke(app, ["transfer", "mrpc-report", "--check"])

    assert result.exit_code == 0, result.output
    assert "metrics NOT reproduced" in result.output
    assert "provide --source PATH" in result.output


def test_mrpc_report_cli_full_check_reproduces_metrics_with_official_source():
    source = Path("data/raw/mrpc/mrpc-test-golden.json")
    if not source.exists():
        pytest.skip("author-only ignored MRPC source is unavailable")

    result = CliRunner().invoke(
        app,
        ["transfer", "mrpc-report", "--check", "--source", str(source)],
    )

    assert result.exit_code == 0, result.output
    assert "OK: M6.1 MRPC metrics reproduced from checksum-verified local source" in result.output


def test_mrpc_report_cli_exits_one_for_missing_source():
    result = CliRunner().invoke(
        app,
        ["transfer", "mrpc-report", "--check", "--source", "/missing/mrpc.json"],
    )

    assert result.exit_code == 1
    assert "VERIFY FAILED: cannot read MRPC source" in result.output


def test_mrpc_report_cli_requires_source_when_writing():
    result = CliRunner().invoke(app, ["transfer", "mrpc-report"])

    assert result.exit_code == 1
    assert "REFUSING: --source is required to generate" in result.output
