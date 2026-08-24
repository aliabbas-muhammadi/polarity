"""Torch-free integrity checks for the frozen xp-006 checkpoint bundle."""

from __future__ import annotations

import hashlib
import json
import math
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from polarity.fingerprint import content_hash
from polarity.release_integrity import ExecutionIdentity, execution_fingerprint

REQUIRED_BUNDLE_FILES = frozenset(
    {
        "added_tokens.json",
        "config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "spm.model",
        "tokenizer.json",
        "tokenizer_config.json",
    }
)
_REQUIRED_MANIFEST_KEYS = frozenset(
    {
        "artifact_bundle",
        "artifact_bundle_fingerprint",
        "calibrator_method",
        "checkpoint_sha256",
        "execution",
        "execution_fingerprint",
        "frozen_thresholds",
        "model_fingerprint",
        "protocol_sha256",
        "recipe_fingerprint",
    }
)
_FROZEN_THRESHOLD_KEYS = frozenset({"zero_fp", "fpr_le_0.01", "fpr_le_0.05"})


class BundleFileDeclaration(BaseModel):
    """One required regular file in a content-addressed checkpoint bundle."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class ArtifactBundleDeclaration(BaseModel):
    """Versioned, fail-closed schema for the complete checkpoint directory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    files: dict[str, BundleFileDeclaration]


@dataclass(frozen=True)
class ArtifactVerificationResult:
    """Successful verification summary for CLI and programmatic callers."""

    files_verified: int
    bundle_fingerprint: str


@dataclass(frozen=True)
class ArtifactReleaseVerification:
    """One manifest snapshot validated as both a complete bundle and scoring context."""

    bundle: ArtifactVerificationResult
    release_context: ArtifactReleaseContext


class ArtifactIdentity(BaseModel):
    """Hashes prediction/report manifests must bind to for one frozen model artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_bundle_fingerprint: str = Field(pattern=r"^pl1:[0-9a-f]{64}$")
    recipe_fingerprint: str = Field(pattern=r"^pl1:[0-9a-f]{64}$")
    execution_fingerprint: str = Field(pattern=r"^pl1:[0-9a-f]{64}$")


class ArtifactReleaseContext(BaseModel):
    """Frozen manifest metadata required before the xp-006 artifact may be scored."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str = Field(min_length=1)
    artifact_identity: ArtifactIdentity
    execution: ExecutionIdentity
    model_revision: str = Field(min_length=1)
    tokenizer_revision: str = Field(min_length=1)
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frozen_thresholds: dict[str, float]
    calibrator_method: Literal["identity", "temperature", "platt", "isotonic"]

    @model_validator(mode="after")
    def revisions_match_execution_identity(self) -> ArtifactReleaseContext:
        if self.model_revision != self.execution.base_revision:
            raise ValueError("model_revision disagrees with execution base_revision")
        if self.tokenizer_revision != self.execution.tokenizer_revision:
            raise ValueError("tokenizer_revision disagrees with execution tokenizer_revision")
        if set(self.frozen_thresholds) != _FROZEN_THRESHOLD_KEYS:
            raise ValueError(
                f"frozen_thresholds must contain exactly {sorted(_FROZEN_THRESHOLD_KEYS)}"
            )
        if any(not math.isfinite(threshold) for threshold in self.frozen_thresholds.values()):
            raise ValueError("frozen thresholds must be finite")
        return self


class ArtifactVerificationError(RuntimeError):
    """Raised when a manifest or bundle cannot satisfy the integrity contract."""


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_artifact_root(artifact_dir: Path) -> None:
    if artifact_dir.is_symlink():
        raise ArtifactVerificationError("artifact root must not be a symlink")
    if not artifact_dir.is_dir():
        raise ArtifactVerificationError(f"artifact root is not a directory: {artifact_dir}")


def _enumerate_bundle_files(artifact_dir: Path) -> dict[str, Path]:
    """Enumerate regular files without following any symlink or special entry."""
    _validate_artifact_root(artifact_dir)
    files: dict[str, Path] = {}

    def visit(directory: Path) -> None:
        try:
            entries = sorted(directory.iterdir(), key=lambda path: path.name)
        except OSError as exc:
            relative = directory.relative_to(artifact_dir).as_posix() or "."
            raise ArtifactVerificationError(
                f"cannot enumerate bundle directory {relative}: {exc}"
            ) from exc
        for entry in entries:
            relative = entry.relative_to(artifact_dir).as_posix()
            try:
                mode = entry.lstat().st_mode
            except OSError as exc:
                raise ArtifactVerificationError(
                    f"cannot inspect bundle path {relative}: {exc}"
                ) from exc
            if stat.S_ISLNK(mode):
                raise ArtifactVerificationError(f"bundle path must not be a symlink: {relative}")
            if stat.S_ISDIR(mode):
                visit(entry)
            elif stat.S_ISREG(mode):
                files[relative] = entry
            else:
                raise ArtifactVerificationError(f"unsupported bundle entry: {relative}")

    visit(artifact_dir)
    return files


def artifact_bundle_fingerprint(bundle: dict[str, Any]) -> str:
    """Content identity for the complete, ordered-independent bundle declaration."""
    return content_hash(bundle)


def _read_manifest_object(manifest_path: Path) -> dict[str, Any]:
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ArtifactVerificationError(
            f"cannot read artifact manifest {manifest_path}: {exc}"
        ) from exc
    try:
        manifest = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ArtifactVerificationError(
            f"malformed artifact manifest JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(manifest, dict):
        raise ArtifactVerificationError("artifact manifest root must be a JSON object")
    return manifest


def _require_manifest_keys(manifest: dict[str, Any]) -> None:
    missing = sorted(_REQUIRED_MANIFEST_KEYS - set(manifest))
    if missing:
        raise ArtifactVerificationError(f"artifact manifest missing required key: {missing[0]}")


def _parse_artifact_identity(manifest: dict[str, Any]) -> ArtifactIdentity:
    try:
        return ArtifactIdentity(
            checkpoint_sha256=manifest["checkpoint_sha256"],
            artifact_bundle_fingerprint=manifest["artifact_bundle_fingerprint"],
            recipe_fingerprint=manifest["recipe_fingerprint"],
            execution_fingerprint=manifest["execution_fingerprint"],
        )
    except ValidationError as exc:
        raise ArtifactVerificationError(f"invalid artifact identity manifest: {exc}") from exc


def build_artifact_bundle(artifact_dir: Path) -> dict[str, Any]:
    """Hash every local bundle file, refusing incomplete or symlinked checkpoint directories."""
    relative = _enumerate_bundle_files(artifact_dir)
    missing = sorted(REQUIRED_BUNDLE_FILES - set(relative))
    if missing:
        raise ArtifactVerificationError(f"missing required bundle file: {missing[0]}")
    files: dict[str, dict[str, int | str]] = {}
    for name, path in relative.items():
        files[name] = {
            "sha256": _sha256_path(path),
            "size_bytes": path.stat().st_size,
        }
    return {"schema_version": 1, "files": files}


def _validate_artifact_identity(manifest: dict[str, Any]) -> ArtifactIdentity:
    """Validate one already-read manifest snapshot and return its four-hash identity."""
    _require_manifest_keys(manifest)
    identity = _parse_artifact_identity(manifest)
    try:
        execution = ExecutionIdentity.model_validate(manifest["execution"])
        bundle = ArtifactBundleDeclaration.model_validate(manifest["artifact_bundle"])
    except (KeyError, ValidationError) as exc:
        raise ArtifactVerificationError(f"invalid artifact identity manifest: {exc}") from exc
    if execution_fingerprint(execution) != identity.execution_fingerprint:
        raise ArtifactVerificationError("execution fingerprint mismatch")
    if artifact_bundle_fingerprint(bundle.model_dump(mode="json")) != (
        identity.artifact_bundle_fingerprint
    ):
        raise ArtifactVerificationError("artifact bundle fingerprint mismatch")
    model_weights = bundle.files.get("model.safetensors")
    if model_weights is None or model_weights.sha256 != identity.checkpoint_sha256:
        raise ArtifactVerificationError(
            "checkpoint_sha256 disagrees with model.safetensors bundle declaration"
        )
    if manifest.get("model_fingerprint") != identity.recipe_fingerprint:
        raise ArtifactVerificationError("model_fingerprint disagrees with recipe_fingerprint")
    return identity


def load_artifact_identity(manifest_path: Path) -> ArtifactIdentity:
    """Load the four-hash binding contract consumed by prediction/report manifests."""
    return _validate_artifact_identity(_read_manifest_object(manifest_path))


def _validate_artifact_release_context(manifest: dict[str, Any]) -> ArtifactReleaseContext:
    """Validate scoring metadata from the same manifest object used for bundle integrity."""
    identity = _validate_artifact_identity(manifest)
    try:
        return ArtifactReleaseContext(
            experiment_id=manifest["experiment_id"],
            artifact_identity=identity,
            execution=ExecutionIdentity.model_validate(manifest["execution"]),
            model_revision=manifest["model_revision"],
            tokenizer_revision=manifest["tokenizer_revision"],
            protocol_sha256=manifest["protocol_sha256"],
            frozen_thresholds=manifest["frozen_thresholds"],
            calibrator_method=manifest["calibrator_method"],
        )
    except (KeyError, ValidationError) as exc:
        raise ArtifactVerificationError(f"invalid artifact release context: {exc}") from exc


def load_artifact_release_context(manifest_path: Path) -> ArtifactReleaseContext:
    """Load the validated model/scoring identity recorded by a frozen artifact manifest."""
    return _validate_artifact_release_context(_read_manifest_object(manifest_path))


def _verify_artifact_bundle(
    manifest: dict[str, Any], artifact_dir: Path
) -> ArtifactVerificationResult:
    """Verify a checkpoint directory against one already-read manifest object."""
    _require_manifest_keys(manifest)
    _parse_artifact_identity(manifest)
    actual_files = set(_enumerate_bundle_files(artifact_dir))
    try:
        execution = ExecutionIdentity.model_validate(manifest["execution"])
    except (KeyError, ValidationError) as exc:
        raise ArtifactVerificationError(f"invalid execution identity: {exc}") from exc
    actual_execution_fingerprint = execution_fingerprint(execution)
    if actual_execution_fingerprint != manifest.get("execution_fingerprint"):
        raise ArtifactVerificationError(
            "execution fingerprint mismatch: "
            f"computed {actual_execution_fingerprint} "
            f"!= manifest {manifest.get('execution_fingerprint')}"
        )
    if manifest.get("model_fingerprint") != manifest.get("recipe_fingerprint"):
        raise ArtifactVerificationError("model_fingerprint disagrees with recipe_fingerprint")
    bundle_raw = manifest["artifact_bundle"]
    if not isinstance(bundle_raw, dict) or bundle_raw.get("schema_version") != 1:
        raise ArtifactVerificationError("artifact bundle schema_version must be 1")
    try:
        bundle_declaration = ArtifactBundleDeclaration.model_validate(bundle_raw)
    except ValidationError as exc:
        raise ArtifactVerificationError(f"invalid artifact bundle manifest: {exc}") from exc
    bundle = bundle_declaration.model_dump(mode="json")
    actual_fingerprint = artifact_bundle_fingerprint(bundle)
    expected_fingerprint = manifest["artifact_bundle_fingerprint"]
    if actual_fingerprint != expected_fingerprint:
        raise ArtifactVerificationError(
            "artifact bundle fingerprint mismatch: "
            f"computed {actual_fingerprint} != manifest {expected_fingerprint}"
        )
    files = bundle_declaration.files
    declared = set(files)
    for name in declared:
        relative = PurePosixPath(name)
        if (
            not name
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != name
            or "\\" in name
        ):
            raise ArtifactVerificationError(f"invalid bundle path: {name}")
    missing_declarations = sorted(REQUIRED_BUNDLE_FILES - declared)
    if missing_declarations:
        raise ArtifactVerificationError(
            f"required bundle declaration missing: {missing_declarations[0]}"
        )
    if manifest.get("checkpoint_sha256") != files["model.safetensors"].sha256:
        raise ArtifactVerificationError(
            "checkpoint_sha256 disagrees with model.safetensors bundle declaration"
        )
    undeclared = sorted(actual_files - declared)
    if undeclared:
        raise ArtifactVerificationError(f"undeclared bundle file: {undeclared[0]}")
    for name, expected in files.items():
        path = artifact_dir / name
        if not path.is_file():
            raise ArtifactVerificationError(f"missing required bundle file: {name}")
        actual_size = path.stat().st_size
        if actual_size != expected.size_bytes:
            raise ArtifactVerificationError(
                f"size mismatch for {name}: actual {actual_size} != manifest {expected.size_bytes}"
            )
        actual = _sha256_path(path)
        if actual != expected.sha256:
            raise ArtifactVerificationError(
                f"sha256 mismatch for {name}: actual {actual} != manifest {expected.sha256}"
            )
    return ArtifactVerificationResult(
        files_verified=len(files),
        bundle_fingerprint=str(manifest["artifact_bundle_fingerprint"]),
    )


def verify_artifact_bundle(manifest_path: Path, artifact_dir: Path) -> ArtifactVerificationResult:
    """Verify a checkpoint bundle against its committed manifest."""
    return _verify_artifact_bundle(_read_manifest_object(manifest_path), artifact_dir)


def verify_artifact_release(manifest_path: Path, artifact_dir: Path) -> ArtifactReleaseVerification:
    """Verify bundle and scoring context from one immutable parsed manifest snapshot."""
    manifest = _read_manifest_object(manifest_path)
    return ArtifactReleaseVerification(
        bundle=_verify_artifact_bundle(manifest, artifact_dir),
        release_context=_validate_artifact_release_context(manifest),
    )


__all__ = [
    "REQUIRED_BUNDLE_FILES",
    "ArtifactBundleDeclaration",
    "ArtifactIdentity",
    "ArtifactReleaseContext",
    "ArtifactReleaseVerification",
    "ArtifactVerificationError",
    "ArtifactVerificationResult",
    "BundleFileDeclaration",
    "artifact_bundle_fingerprint",
    "build_artifact_bundle",
    "load_artifact_identity",
    "load_artifact_release_context",
    "verify_artifact_bundle",
    "verify_artifact_release",
]
