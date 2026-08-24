import builtins
import hashlib
import json
import math
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

import polarity.artifact as artifact_mod
import polarity.cli as cli_mod
import polarity.transfer as transfer_mod
from polarity import release_integrity
from polarity.artifact import (
    ArtifactIdentity,
    ArtifactVerificationError,
    artifact_bundle_fingerprint,
    build_artifact_bundle,
    load_artifact_identity,
    load_artifact_release_context,
    verify_artifact_bundle,
)
from polarity.cli import app
from polarity.train import config as trainconfig

REQUIRED_FILES = (
    "added_tokens.json",
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "spm.model",
    "tokenizer.json",
    "tokenizer_config.json",
)
XP006_MANIFEST = Path("reports/m6/model-manifest-xp-006-deberta-transformers-fixed.json")


def _bundle_fingerprint(bundle: object) -> str:
    canonical = json.dumps(
        bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()
    return f"pl1:{hashlib.sha256(canonical).hexdigest()}"


def _execution_fingerprint(identity: object) -> str:
    payload = {"identity_kind": "execution", "identity": identity}
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()
    return f"pl1:{hashlib.sha256(canonical).hexdigest()}"


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_complete_bundle(tmp_path: Path) -> tuple[Path, Path, str]:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    files: dict[str, dict[str, int | str]] = {}
    for index, name in enumerate(REQUIRED_FILES):
        content = f"fixture:{index}:{name}".encode()
        (artifact_dir / name).write_bytes(content)
        files[name] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }
    bundle = {"schema_version": 1, "files": files}
    fingerprint = _bundle_fingerprint(bundle)
    execution = {
        "schema_version": 1,
        "python": "3.13.14",
        "platform": "test-platform",
        "torch": "2.6.0+cu124",
        "transformers": "4.57.6",
        "cuda": "12.4",
        "base_revision": "base-sha",
        "tokenizer_revision": "tokenizer-sha",
    }
    recipe_fingerprint = f"pl1:{'1' * 64}"
    manifest = {
        "artifact_bundle": bundle,
        "artifact_bundle_fingerprint": fingerprint,
        "checkpoint_sha256": files["model.safetensors"]["sha256"],
        "execution": execution,
        "execution_fingerprint": _execution_fingerprint(execution),
        "model_fingerprint": recipe_fingerprint,
        "recipe_fingerprint": recipe_fingerprint,
        "experiment_id": "xp-fixture",
        "model_revision": execution["base_revision"],
        "tokenizer_revision": execution["tokenizer_revision"],
        "protocol_sha256": "0" * 64,
        "frozen_thresholds": {
            "zero_fp": 0.5,
            "fpr_le_0.01": 0.5,
            "fpr_le_0.05": 0.5,
        },
        "calibrator_method": "identity",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, artifact_dir, fingerprint


def _write_bound_scoring_context(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    config_path = Path("configs/train/m3-3-deberta-fixed.yaml").resolve()
    cfg = trainconfig.load_config(config_path)
    manifest_path, artifact_dir, _ = _write_complete_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["experiment_id"] = cfg.experiment_id
    manifest["recipe_fingerprint"] = trainconfig.recipe_fingerprint(cfg)
    manifest["model_fingerprint"] = manifest["recipe_fingerprint"]
    manifest["model_revision"] = cfg.model.revision
    manifest["tokenizer_revision"] = cfg.model.tokenizer_revision
    manifest["execution"]["base_revision"] = cfg.model.revision
    manifest["execution"]["tokenizer_revision"] = cfg.model.tokenizer_revision
    manifest["execution_fingerprint"] = _execution_fingerprint(manifest["execution"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    protocol = json.loads(
        Path("reports/m6/protocol-xp-006-deberta-transformers-fixed.json").read_text(
            encoding="utf-8"
        )
    )
    protocol["experiment_id"] = cfg.experiment_id
    for field in (
        "checkpoint_sha256",
        "artifact_bundle_fingerprint",
        "recipe_fingerprint",
        "execution_fingerprint",
    ):
        protocol[field] = manifest[field]
    protocol["model_fingerprint"] = manifest["recipe_fingerprint"]
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    manifest["protocol_sha256"] = _sha256_path(protocol_path)
    manifest["frozen_thresholds"] = {
        key: point["threshold"] for key, point in protocol["operating_points"].items()
    }
    manifest["calibrator_method"] = protocol["calibrator_method"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, artifact_dir, protocol_path, config_path


def _write_mutated_config(tmp_path: Path, field: str) -> Path:
    cfg = trainconfig.load_config(Path("configs/train/m3-3-deberta-fixed.yaml"))
    payload = cfg.model_dump(mode="json")
    if field == "experiment_id":
        payload["experiment_id"] = "xp-wrong"
    elif field == "recipe":
        payload["training"]["learning_rate"] = 3e-5
    elif field == "base_revision":
        payload["model"]["revision"] = "wrong-base-revision"
    elif field == "tokenizer_revision":
        payload["model"]["tokenizer_revision"] = "wrong-tokenizer-revision"
    elif field == "max_length":
        payload["training"]["max_length"] = 256
    else:
        raise AssertionError(f"unhandled mutation: {field}")
    config_path = tmp_path / f"config-{field}.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def test_verify_artifact_bundle_accepts_complete_declared_bundle(tmp_path: Path):
    manifest, artifact_dir, expected_fingerprint = _write_complete_bundle(tmp_path)

    result = verify_artifact_bundle(manifest, artifact_dir)

    assert result.files_verified == 7
    assert result.bundle_fingerprint == expected_fingerprint


def test_build_artifact_bundle_hashes_every_required_file_deterministically(tmp_path: Path):
    _, artifact_dir, expected_fingerprint = _write_complete_bundle(tmp_path)

    first = build_artifact_bundle(artifact_dir)
    second = build_artifact_bundle(artifact_dir)

    assert first == second
    assert set(first["files"]) == set(REQUIRED_FILES)
    assert artifact_bundle_fingerprint(first) == expected_fingerprint


def test_verify_artifact_bundle_rejects_missing_file(tmp_path: Path):
    manifest, artifact_dir, _ = _write_complete_bundle(tmp_path)
    (artifact_dir / "spm.model").unlink()

    with pytest.raises(ArtifactVerificationError, match=r"missing required bundle file: spm.model"):
        verify_artifact_bundle(manifest, artifact_dir)


def test_verify_artifact_bundle_rejects_hash_mismatch(tmp_path: Path):
    manifest, artifact_dir, _ = _write_complete_bundle(tmp_path)
    path = artifact_dir / "config.json"
    path.write_bytes(b"x" * path.stat().st_size)

    with pytest.raises(ArtifactVerificationError, match=r"sha256 mismatch for config.json"):
        verify_artifact_bundle(manifest, artifact_dir)


def test_verify_artifact_bundle_rejects_size_mismatch(tmp_path: Path):
    manifest_path, artifact_dir, _ = _write_complete_bundle(tmp_path)
    path = artifact_dir / "config.json"
    path.write_bytes(path.read_bytes() + b"x")

    with pytest.raises(ArtifactVerificationError, match=r"size mismatch for config.json"):
        verify_artifact_bundle(manifest_path, artifact_dir)


def test_verify_artifact_bundle_rejects_undeclared_file(tmp_path: Path):
    manifest, artifact_dir, _ = _write_complete_bundle(tmp_path)
    (artifact_dir / "training_args.bin").write_bytes(b"undeclared")

    with pytest.raises(
        ArtifactVerificationError, match=r"undeclared bundle file: training_args.bin"
    ):
        verify_artifact_bundle(manifest, artifact_dir)


def test_verify_artifact_bundle_rejects_omitted_required_declaration(tmp_path: Path):
    manifest_path, artifact_dir, _ = _write_complete_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["artifact_bundle"]["files"]["tokenizer.json"]
    manifest["artifact_bundle_fingerprint"] = _bundle_fingerprint(manifest["artifact_bundle"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (artifact_dir / "tokenizer.json").unlink()

    with pytest.raises(
        ArtifactVerificationError,
        match=r"required bundle declaration missing: tokenizer.json",
    ):
        verify_artifact_bundle(manifest_path, artifact_dir)


def test_verify_artifact_bundle_rejects_manifest_fingerprint_mismatch(tmp_path: Path):
    manifest_path, artifact_dir, _ = _write_complete_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_bundle_fingerprint"] = f"pl1:{'0' * 64}"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactVerificationError, match="artifact bundle fingerprint mismatch"):
        verify_artifact_bundle(manifest_path, artifact_dir)


def test_verify_artifact_bundle_rejects_execution_fingerprint_mismatch(tmp_path: Path):
    manifest_path, artifact_dir, _ = _write_complete_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["execution"]["transformers"] = "5.15.1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactVerificationError, match="execution fingerprint mismatch"):
        verify_artifact_bundle(manifest_path, artifact_dir)


def test_verify_artifact_bundle_rejects_legacy_recipe_alias_disagreement(tmp_path: Path):
    manifest_path, artifact_dir, _ = _write_complete_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_fingerprint"] = f"pl1:{'2' * 64}"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ArtifactVerificationError,
        match="model_fingerprint disagrees with recipe_fingerprint",
    ):
        verify_artifact_bundle(manifest_path, artifact_dir)


def test_verify_artifact_bundle_rejects_path_outside_artifact_dir(tmp_path: Path):
    manifest_path, artifact_dir, _ = _write_complete_bundle(tmp_path)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_bundle"]["files"]["../outside.bin"] = {
        "sha256": hashlib.sha256(b"outside").hexdigest(),
        "size_bytes": len(b"outside"),
    }
    manifest["artifact_bundle_fingerprint"] = _bundle_fingerprint(manifest["artifact_bundle"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactVerificationError, match=r"invalid bundle path: ../outside.bin"):
        verify_artifact_bundle(manifest_path, artifact_dir)


def test_verify_artifact_bundle_rejects_unknown_schema_version(tmp_path: Path):
    manifest_path, artifact_dir, _ = _write_complete_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_bundle"]["schema_version"] = 2
    manifest["artifact_bundle_fingerprint"] = _bundle_fingerprint(manifest["artifact_bundle"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactVerificationError, match="artifact bundle schema_version must be 1"):
        verify_artifact_bundle(manifest_path, artifact_dir)


def test_verify_artifact_bundle_rejects_malformed_file_declaration(tmp_path: Path):
    manifest_path, artifact_dir, _ = _write_complete_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["artifact_bundle"]["files"]["config.json"]["sha256"]
    manifest["artifact_bundle_fingerprint"] = _bundle_fingerprint(manifest["artifact_bundle"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactVerificationError, match="invalid artifact bundle manifest"):
        verify_artifact_bundle(manifest_path, artifact_dir)


def test_verify_artifact_bundle_rejects_legacy_checkpoint_hash_disagreement(tmp_path: Path):
    manifest_path, artifact_dir, _ = _write_complete_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoint_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ArtifactVerificationError,
        match=r"checkpoint_sha256 disagrees with model.safetensors",
    ):
        verify_artifact_bundle(manifest_path, artifact_dir)


def test_verify_artifact_bundle_rejects_symlinked_required_file(tmp_path: Path):
    manifest, artifact_dir, _ = _write_complete_bundle(tmp_path)
    path = artifact_dir / "spm.model"
    target = tmp_path / "outside-spm.model"
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)

    with pytest.raises(
        ArtifactVerificationError,
        match=r"bundle path must not be a symlink: spm.model",
    ):
        verify_artifact_bundle(manifest, artifact_dir)


def test_build_and_verify_reject_symlinked_artifact_root(tmp_path: Path):
    manifest, artifact_dir, _ = _write_complete_bundle(tmp_path)
    symlinked_root = tmp_path / "artifact-link"
    symlinked_root.symlink_to(artifact_dir, target_is_directory=True)

    with pytest.raises(
        ArtifactVerificationError,
        match=r"artifact root must not be a symlink",
    ):
        build_artifact_bundle(symlinked_root)
    with pytest.raises(
        ArtifactVerificationError,
        match=r"artifact root must not be a symlink",
    ):
        verify_artifact_bundle(manifest, symlinked_root)


def test_build_and_verify_reject_nested_parent_symlink(tmp_path: Path):
    manifest_path, artifact_dir, _ = _write_complete_bundle(tmp_path)
    nested_target = tmp_path / "nested-target"
    nested_target.mkdir()
    nested_content = b"nested"
    (nested_target / "extra.bin").write_bytes(nested_content)
    (artifact_dir / "nested").symlink_to(nested_target, target_is_directory=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_bundle"]["files"]["nested/extra.bin"] = {
        "sha256": hashlib.sha256(nested_content).hexdigest(),
        "size_bytes": len(nested_content),
    }
    manifest["artifact_bundle_fingerprint"] = _bundle_fingerprint(manifest["artifact_bundle"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ArtifactVerificationError,
        match=r"bundle path must not be a symlink: nested",
    ):
        build_artifact_bundle(artifact_dir)
    with pytest.raises(
        ArtifactVerificationError,
        match=r"bundle path must not be a symlink: nested",
    ):
        verify_artifact_bundle(manifest_path, artifact_dir)


def test_build_and_verify_reject_dangling_symlink_entry(tmp_path: Path):
    manifest, artifact_dir, _ = _write_complete_bundle(tmp_path)
    (artifact_dir / "dangling.bin").symlink_to(tmp_path / "missing-target.bin")

    with pytest.raises(
        ArtifactVerificationError,
        match=r"bundle path must not be a symlink: dangling.bin",
    ):
        build_artifact_bundle(artifact_dir)
    with pytest.raises(
        ArtifactVerificationError,
        match=r"bundle path must not be a symlink: dangling.bin",
    ):
        verify_artifact_bundle(manifest, artifact_dir)


def test_build_and_verify_reject_undeclared_symlink_entry(tmp_path: Path):
    manifest, artifact_dir, _ = _write_complete_bundle(tmp_path)
    (artifact_dir / "undeclared-link.bin").symlink_to(artifact_dir / "config.json")

    with pytest.raises(
        ArtifactVerificationError,
        match=r"bundle path must not be a symlink: undeclared-link.bin",
    ):
        build_artifact_bundle(artifact_dir)
    with pytest.raises(
        ArtifactVerificationError,
        match=r"bundle path must not be a symlink: undeclared-link.bin",
    ):
        verify_artifact_bundle(manifest, artifact_dir)


def test_artifact_verify_cli_reports_success(tmp_path: Path):
    manifest, artifact_dir, fingerprint = _write_complete_bundle(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "artifact",
            "verify",
            "--manifest",
            str(manifest),
            "--artifact-dir",
            str(artifact_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert f"OK: verified 7 bundle files ({fingerprint})" in result.output


def test_artifact_verify_cli_exits_one_with_integrity_error(tmp_path: Path):
    manifest, artifact_dir, _ = _write_complete_bundle(tmp_path)
    (artifact_dir / "config.json").unlink()

    result = CliRunner().invoke(
        app,
        [
            "artifact",
            "verify",
            "--manifest",
            str(manifest),
            "--artifact-dir",
            str(artifact_dir),
        ],
    )

    assert result.exit_code == 1
    assert "VERIFY FAILED: missing required bundle file: config.json" in result.output


def test_artifact_verify_cli_normalizes_missing_manifest(tmp_path: Path):
    _, artifact_dir, _ = _write_complete_bundle(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "artifact",
            "verify",
            "--manifest",
            str(tmp_path / "missing.json"),
            "--artifact-dir",
            str(artifact_dir),
        ],
    )

    assert result.exit_code == 1
    assert "VERIFY FAILED: cannot read artifact manifest" in result.output


def test_artifact_verify_cli_normalizes_unreadable_manifest_path(tmp_path: Path):
    _, artifact_dir, _ = _write_complete_bundle(tmp_path)
    manifest_dir = tmp_path / "manifest-directory"
    manifest_dir.mkdir()

    result = CliRunner().invoke(
        app,
        [
            "artifact",
            "verify",
            "--manifest",
            str(manifest_dir),
            "--artifact-dir",
            str(artifact_dir),
        ],
    )

    assert result.exit_code == 1
    assert "VERIFY FAILED: cannot read artifact manifest" in result.output


def test_artifact_verify_cli_normalizes_malformed_manifest_json(tmp_path: Path):
    _, artifact_dir, _ = _write_complete_bundle(tmp_path)
    manifest = tmp_path / "malformed.json"
    manifest.write_text("{not-json", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "artifact",
            "verify",
            "--manifest",
            str(manifest),
            "--artifact-dir",
            str(artifact_dir),
        ],
    )

    assert result.exit_code == 1
    assert "VERIFY FAILED: malformed artifact manifest JSON" in result.output


def test_artifact_verify_cli_normalizes_non_object_manifest(tmp_path: Path):
    _, artifact_dir, _ = _write_complete_bundle(tmp_path)
    manifest = tmp_path / "non-object.json"
    manifest.write_text("[]", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "artifact",
            "verify",
            "--manifest",
            str(manifest),
            "--artifact-dir",
            str(artifact_dir),
        ],
    )

    assert result.exit_code == 1
    assert "VERIFY FAILED: artifact manifest root must be a JSON object" in result.output


def test_artifact_verify_cli_normalizes_missing_manifest_key(tmp_path: Path):
    manifest_path, artifact_dir, _ = _write_complete_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["artifact_bundle"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "artifact",
            "verify",
            "--manifest",
            str(manifest_path),
            "--artifact-dir",
            str(artifact_dir),
        ],
    )

    assert result.exit_code == 1
    assert "VERIFY FAILED: artifact manifest missing required key: artifact_bundle" in result.output


def test_artifact_verify_cli_rejects_malformed_identity_hash_shape(tmp_path: Path):
    manifest_path, artifact_dir, _ = _write_complete_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_fingerprint"] = "pl1:recipe"
    manifest["recipe_fingerprint"] = "pl1:recipe"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "artifact",
            "verify",
            "--manifest",
            str(manifest_path),
            "--artifact-dir",
            str(artifact_dir),
        ],
    )

    assert result.exit_code == 1
    assert "VERIFY FAILED: invalid artifact identity manifest" in result.output


def test_committed_xp006_manifest_binds_complete_bundle_and_dual_identities():
    manifest = json.loads(XP006_MANIFEST.read_text(encoding="utf-8"))
    protocol_path = Path("reports/m6/protocol-xp-006-deberta-transformers-fixed.json")

    assert set(manifest["artifact_bundle"]["files"]) == set(REQUIRED_FILES)
    assert manifest["artifact_bundle_fingerprint"] == (
        "pl1:0dfcb787d1b5ea811b7405606f8a1e1ae95d50802053c0227324d16cb3c33df1"
    )
    assert manifest["recipe_fingerprint"] == (
        "pl1:04857dd3f1522863cc9e99378569403ab76f72f5f64d7fc59abc98e38a125c17"
    )
    assert manifest["execution_fingerprint"] == (
        "pl1:bc55b62acce7235c6442b53e2b3cf1147241619cac801899d7993db7ee289cec"
    )
    assert manifest["execution"]["python"] == "unknown:not-recorded"
    assert manifest["execution"]["platform"] == "unknown:not-recorded"
    assert manifest["execution"]["cuda"] == "unknown:not-recorded"
    assert manifest["protocol_sha256"] == _sha256_path(protocol_path)


def test_load_artifact_identity_exposes_hash_contract_for_prediction_manifests():
    assert load_artifact_identity(XP006_MANIFEST) == ArtifactIdentity(
        checkpoint_sha256="8d765a7e55c94bb5a8c7754eb705ddf8dc7e6c74cc72e1b0ce4c9110e7a0f355",
        artifact_bundle_fingerprint=(
            "pl1:0dfcb787d1b5ea811b7405606f8a1e1ae95d50802053c0227324d16cb3c33df1"
        ),
        recipe_fingerprint=("pl1:04857dd3f1522863cc9e99378569403ab76f72f5f64d7fc59abc98e38a125c17"),
        execution_fingerprint=(
            "pl1:bc55b62acce7235c6442b53e2b3cf1147241619cac801899d7993db7ee289cec"
        ),
    )


def test_load_artifact_release_context_validates_one_manifest_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest_path, _, _ = _write_complete_bundle(tmp_path)
    original_reader = artifact_mod._read_manifest_object
    reads = 0

    def read_once(path: Path):
        nonlocal reads
        reads += 1
        if reads > 1:
            raise AssertionError("release context reread a mutable manifest")
        return original_reader(path)

    monkeypatch.setattr(artifact_mod, "_read_manifest_object", read_once)

    context = load_artifact_release_context(manifest_path)

    assert reads == 1
    assert context.experiment_id == "xp-fixture"


@pytest.mark.parametrize(
    ("field", "malformed"),
    [
        ("checkpoint_sha256", "deadbeef"),
        ("artifact_bundle_fingerprint", "pl1:bundle"),
        ("recipe_fingerprint", "pl1:recipe"),
        ("execution_fingerprint", "pl1:execution"),
    ],
)
def test_artifact_identity_rejects_malformed_hash_shapes(field: str, malformed: str):
    values = {
        "checkpoint_sha256": "a" * 64,
        "artifact_bundle_fingerprint": f"pl1:{'b' * 64}",
        "recipe_fingerprint": f"pl1:{'c' * 64}",
        "execution_fingerprint": f"pl1:{'d' * 64}",
    }
    values[field] = malformed

    with pytest.raises(ValidationError):
        ArtifactIdentity(**values)


def test_artifact_verifier_does_not_import_torch_or_model_code(tmp_path: Path, monkeypatch):
    manifest, artifact_dir, _ = _write_complete_bundle(tmp_path)
    original_import = builtins.__import__

    def refuse_model_runtime(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch.") or name == "transformers":
            raise AssertionError(f"artifact verifier imported model runtime: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse_model_runtime)

    assert verify_artifact_bundle(manifest, artifact_dir).files_verified == 7


def test_transfer_veritas_refuses_tampered_bundle_before_scoring(tmp_path: Path, monkeypatch):
    manifest_path, artifact_dir, _ = _write_complete_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = json.loads(
        Path("reports/m6/protocol-xp-006-deberta-transformers-fixed.json").read_text(
            encoding="utf-8"
        )
    )
    protocol["checkpoint_sha256"] = manifest["checkpoint_sha256"]
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    prediction_path = tmp_path / "scores.jsonl"
    tokenizer = artifact_dir / "tokenizer.json"
    tokenizer.write_bytes(b"x" * tokenizer.stat().st_size)
    scored = False

    def fail_if_scored(*args, **kwargs):
        nonlocal scored
        scored = True
        return []

    monkeypatch.setattr(cli_mod, "M6_MODEL_MANIFEST", manifest_path, raising=False)
    monkeypatch.setattr(cli_mod, "M6_PROTOCOL", protocol_path)
    monkeypatch.setattr(cli_mod, "M6_VERITAS_PRED", prediction_path)
    monkeypatch.setattr(transfer_mod, "score_golden", fail_if_scored)

    result = CliRunner().invoke(app, ["transfer", "veritas", str(artifact_dir)])

    assert result.exit_code == 1
    assert "REFUSING: artifact bundle verification failed" in result.output
    assert scored is False


@pytest.mark.parametrize(
    "mismatched_field",
    [
        "artifact_bundle_fingerprint",
        "recipe_fingerprint",
        "execution_fingerprint",
    ],
)
def test_transfer_veritas_refuses_protocol_artifact_identity_mismatch_before_scoring(
    tmp_path: Path, monkeypatch, mismatched_field: str
):
    manifest_path, artifact_dir, _ = _write_complete_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = json.loads(
        Path("reports/m6/protocol-xp-006-deberta-transformers-fixed.json").read_text(
            encoding="utf-8"
        )
    )
    for field in (
        "checkpoint_sha256",
        "artifact_bundle_fingerprint",
        "recipe_fingerprint",
        "execution_fingerprint",
    ):
        protocol[field] = manifest[field]
    protocol["model_fingerprint"] = manifest["recipe_fingerprint"]
    protocol[mismatched_field] = f"pl1:{'e' * 64}"
    if mismatched_field == "recipe_fingerprint":
        protocol["model_fingerprint"] = protocol[mismatched_field]
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    manifest["protocol_sha256"] = _sha256_path(protocol_path)
    manifest["frozen_thresholds"] = {
        key: point["threshold"] for key, point in protocol["operating_points"].items()
    }
    manifest["calibrator_method"] = protocol["calibrator_method"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    scored = False

    def fail_if_scored(*args, **kwargs):
        nonlocal scored
        scored = True
        return []

    monkeypatch.setattr(cli_mod, "M6_MODEL_MANIFEST", manifest_path, raising=False)
    monkeypatch.setattr(cli_mod, "M6_PROTOCOL", protocol_path)
    monkeypatch.setattr(cli_mod, "M6_VERITAS_PRED", tmp_path / "scores.jsonl")
    monkeypatch.setattr(transfer_mod, "score_golden", fail_if_scored)

    result = CliRunner().invoke(app, ["transfer", "veritas", str(artifact_dir)])

    assert result.exit_code == 1
    assert "frozen protocol identity does not match the artifact manifest" in result.output
    assert scored is False


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("experiment_id", "experiment_id"),
        ("recipe", "recipe fingerprint"),
        ("base_revision", "base revision"),
        ("tokenizer_revision", "tokenizer revision"),
        ("max_length", "recipe fingerprint"),
    ],
)
def test_transfer_veritas_refuses_scoring_config_mismatch_before_runtime_or_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    expected: str,
):
    manifest_path, artifact_dir, protocol_path, _ = _write_bound_scoring_context(tmp_path)
    config_path = _write_mutated_config(tmp_path, field)
    runtime_checked = False
    scored = False

    def runtime_guard() -> dict[str, str]:
        nonlocal runtime_checked
        runtime_checked = True
        return {"torch": "2.6.0+cu124", "transformers": "4.57.6"}

    def fail_if_scored(*args, **kwargs):
        nonlocal scored
        scored = True
        return []

    monkeypatch.setattr(cli_mod, "M6_MODEL_MANIFEST", manifest_path)
    monkeypatch.setattr(cli_mod, "M6_PROTOCOL", protocol_path)
    monkeypatch.setattr(cli_mod, "M6_VERITAS_PRED", tmp_path / "scores.jsonl")
    monkeypatch.setattr(release_integrity, "assert_installed_xp006_release_profile", runtime_guard)
    monkeypatch.setattr(transfer_mod, "score_golden", fail_if_scored)

    result = CliRunner().invoke(
        app,
        ["transfer", "veritas", str(artifact_dir), "--config", str(config_path)],
    )

    assert result.exit_code == 1
    assert f"REFUSING: {expected}" in result.output
    assert runtime_checked is False
    assert scored is False


@pytest.mark.parametrize("mutation", ["threshold", "method", "same_method_parameters"])
def test_transfer_veritas_refuses_protocol_mutation_before_runtime_or_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
):
    manifest_path, artifact_dir, protocol_path, config_path = _write_bound_scoring_context(tmp_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if mutation == "threshold":
        protocol["operating_points"]["fpr_le_0.01"]["threshold"] = 0.994834
    elif mutation == "method":
        protocol["calibrator_method"] = "identity"
        protocol["calibrator_params"] = {}
    elif mutation == "same_method_parameters":
        protocol["calibrator_params"]["iso_y"][0] = math.nextafter(0.0, 1.0)
    else:
        raise AssertionError(f"unhandled protocol mutation: {mutation}")
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    runtime_checked = False
    scored = False

    def runtime_guard() -> dict[str, str]:
        nonlocal runtime_checked
        runtime_checked = True
        return {"torch": "2.6.0+cu124", "transformers": "4.57.6"}

    def fail_if_scored(*args, **kwargs):
        nonlocal scored
        scored = True
        return []

    monkeypatch.setattr(cli_mod, "M6_MODEL_MANIFEST", manifest_path)
    monkeypatch.setattr(cli_mod, "M6_PROTOCOL", protocol_path)
    monkeypatch.setattr(cli_mod, "M6_VERITAS_PRED", tmp_path / "scores.jsonl")
    monkeypatch.setattr(release_integrity, "assert_installed_xp006_release_profile", runtime_guard)
    monkeypatch.setattr(transfer_mod, "score_golden", fail_if_scored)

    result = CliRunner().invoke(
        app,
        ["transfer", "veritas", str(artifact_dir), "--config", str(config_path)],
    )

    assert result.exit_code == 1
    assert "protocol sha256 does not match the artifact manifest" in result.output
    assert runtime_checked is False
    assert scored is False


@pytest.mark.parametrize("manifest_mutation", ["threshold", "method"])
def test_transfer_veritas_refuses_manifest_scoring_contract_mismatch_before_runtime_or_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest_mutation: str,
):
    manifest_path, artifact_dir, protocol_path, config_path = _write_bound_scoring_context(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest_mutation == "threshold":
        manifest["frozen_thresholds"]["fpr_le_0.01"] = 0.994834
    else:
        manifest["calibrator_method"] = "identity"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    runtime_checked = False
    scored = False

    def runtime_guard() -> dict[str, str]:
        nonlocal runtime_checked
        runtime_checked = True
        return {"torch": "2.6.0+cu124", "transformers": "4.57.6"}

    def fail_if_scored(*args, **kwargs):
        nonlocal scored
        scored = True
        return []

    monkeypatch.setattr(cli_mod, "M6_MODEL_MANIFEST", manifest_path)
    monkeypatch.setattr(cli_mod, "M6_PROTOCOL", protocol_path)
    monkeypatch.setattr(cli_mod, "M6_VERITAS_PRED", tmp_path / "scores.jsonl")
    monkeypatch.setattr(release_integrity, "assert_installed_xp006_release_profile", runtime_guard)
    monkeypatch.setattr(transfer_mod, "score_golden", fail_if_scored)

    result = CliRunner().invoke(
        app,
        ["transfer", "veritas", str(artifact_dir), "--config", str(config_path)],
    )

    assert result.exit_code == 1
    assert f"{manifest_mutation} does not match the frozen protocol" in result.output
    assert runtime_checked is False
    assert scored is False


@pytest.mark.parametrize("config_state", ["missing", "malformed", "invalid_utf8"])
def test_transfer_veritas_normalizes_invalid_config_before_runtime_or_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_state: str
):
    manifest_path, artifact_dir, protocol_path, _ = _write_bound_scoring_context(tmp_path)
    config_path = tmp_path / "requested.yaml"
    if config_state == "malformed":
        config_path.write_text("model: [\n", encoding="utf-8")
    elif config_state == "invalid_utf8":
        config_path.write_bytes(b"\xff")
    scored = False

    def fail_if_scored(*args, **kwargs):
        nonlocal scored
        scored = True
        return []

    monkeypatch.setattr(cli_mod, "M6_MODEL_MANIFEST", manifest_path)
    monkeypatch.setattr(cli_mod, "M6_PROTOCOL", protocol_path)
    monkeypatch.setattr(cli_mod, "M6_VERITAS_PRED", tmp_path / "scores.jsonl")
    monkeypatch.setattr(transfer_mod, "score_golden", fail_if_scored)

    result = CliRunner().invoke(
        app,
        ["transfer", "veritas", str(artifact_dir), "--config", str(config_path)],
    )

    assert result.exit_code == 1
    assert "REFUSING: invalid scoring config" in result.output
    assert "Traceback" not in result.output
    assert scored is False


def test_transfer_veritas_refuses_release_runtime_mismatch_before_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest_path, artifact_dir, protocol_path, config_path = _write_bound_scoring_context(tmp_path)
    scored = False

    def reject_runtime() -> dict[str, str]:
        raise release_integrity.ReleaseProfileError("installed Transformers 5.15.1 is rejected")

    def fail_if_scored(*args, **kwargs):
        nonlocal scored
        scored = True
        return []

    monkeypatch.setattr(cli_mod, "M6_MODEL_MANIFEST", manifest_path)
    monkeypatch.setattr(cli_mod, "M6_PROTOCOL", protocol_path)
    monkeypatch.setattr(cli_mod, "M6_VERITAS_PRED", tmp_path / "scores.jsonl")
    monkeypatch.setattr(release_integrity, "assert_installed_xp006_release_profile", reject_runtime)
    monkeypatch.setattr(transfer_mod, "score_golden", fail_if_scored)

    result = CliRunner().invoke(
        app,
        ["transfer", "veritas", str(artifact_dir), "--config", str(config_path)],
    )

    assert result.exit_code == 1
    assert "REFUSING: xp-006 release profile mismatch" in result.output
    assert scored is False


def test_transfer_veritas_validates_bound_context_before_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest_path, artifact_dir, protocol_path, config_path = _write_bound_scoring_context(tmp_path)
    prediction_path = tmp_path / "scores.jsonl"
    runtime_checked = False
    calibrator_built = False
    scored_config: trainconfig.TrainConfig | None = None

    original_calibrator = transfer_mod._calibrator

    def record_calibrator(protocol):
        nonlocal calibrator_built
        assert runtime_checked is False
        calibrator_built = True
        return original_calibrator(protocol)

    def runtime_guard() -> dict[str, str]:
        nonlocal runtime_checked
        runtime_checked = True
        return {"torch": "2.6.0+cu124", "transformers": "4.57.6"}

    def record_score(_checkpoint, cfg, calibrator, *, device):
        nonlocal scored_config
        assert runtime_checked is True
        assert calibrator_built is True
        assert calibrator.method == "isotonic"
        assert device == "cpu"
        scored_config = cfg
        return []

    monkeypatch.setattr(cli_mod, "M6_MODEL_MANIFEST", manifest_path)
    monkeypatch.setattr(cli_mod, "M6_PROTOCOL", protocol_path)
    monkeypatch.setattr(cli_mod, "M6_VERITAS_PRED", prediction_path)
    monkeypatch.setattr(transfer_mod, "_calibrator", record_calibrator)
    monkeypatch.setattr(release_integrity, "assert_installed_xp006_release_profile", runtime_guard)
    monkeypatch.setattr(transfer_mod, "score_golden", record_score)

    result = CliRunner().invoke(
        app,
        ["transfer", "veritas", str(artifact_dir), "--config", str(config_path)],
    )

    assert result.exit_code == 0
    assert isinstance(scored_config, trainconfig.TrainConfig)
    assert scored_config.training.max_length == 128


def test_transfer_veritas_reads_manifest_once_for_bundle_and_release_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest_path, artifact_dir, protocol_path, config_path = _write_bound_scoring_context(tmp_path)
    prediction_path = tmp_path / "scores.jsonl"
    original_reader = artifact_mod._read_manifest_object
    manifest_reads = 0
    scored = False

    def count_manifest_reads(path: Path):
        nonlocal manifest_reads
        if path == manifest_path:
            manifest_reads += 1
        return original_reader(path)

    def runtime_guard() -> dict[str, str]:
        return {"torch": "2.6.0+cu124", "transformers": "4.57.6"}

    def record_score(*args, **kwargs):
        nonlocal scored
        assert manifest_reads == 1
        scored = True
        return []

    monkeypatch.setattr(cli_mod, "M6_MODEL_MANIFEST", manifest_path)
    monkeypatch.setattr(cli_mod, "M6_PROTOCOL", protocol_path)
    monkeypatch.setattr(cli_mod, "M6_VERITAS_PRED", prediction_path)
    monkeypatch.setattr(artifact_mod, "_read_manifest_object", count_manifest_reads)
    monkeypatch.setattr(release_integrity, "assert_installed_xp006_release_profile", runtime_guard)
    monkeypatch.setattr(transfer_mod, "score_golden", record_score)

    result = CliRunner().invoke(
        app,
        ["transfer", "veritas", str(artifact_dir), "--config", str(config_path)],
    )

    assert result.exit_code == 0, result.output
    assert manifest_reads == 1
    assert scored is True
