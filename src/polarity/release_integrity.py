"""Release-only environment identity and dependency guards.

This module is deliberately Torch-free. It records version strings supplied by committed
provenance or package metadata; it never imports model runtimes.
"""

from __future__ import annotations

import re
from importlib import metadata
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from polarity.fingerprint import content_hash

UNKNOWN_EXECUTION_VALUE = "unknown:not-recorded"
XP006_TRANSFORMERS_VERSION = "4.57.6"
XP006_TORCH_MIN = (2, 6)
XP006_TORCH_MAX_EXCLUSIVE = (2, 7)


class ReleaseProfileError(RuntimeError):
    """Raised when a runtime cannot reproduce the frozen xp-006 release profile."""


def _major_minor(version: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"(\d+)\.(\d+)(?:\.\d+)?(?:\+[0-9A-Za-z.]+)?", version)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


class ExecutionIdentity(BaseModel):
    """Exact recorded inputs that distinguish one execution environment from another."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    python: str
    platform: str
    torch: str
    transformers: str
    cuda: str
    base_revision: str
    tokenizer_revision: str

    @field_validator(
        "python",
        "platform",
        "torch",
        "transformers",
        "cuda",
        "base_revision",
        "tokenizer_revision",
    )
    @classmethod
    def values_are_explicit(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                f"execution identity values must be explicit; use {UNKNOWN_EXECUTION_VALUE!r} "
                "when historical telemetry was not recorded"
            )
        return value


def execution_fingerprint(identity: ExecutionIdentity) -> str:
    """Return a deterministic identity for the exact recorded execution environment."""
    return content_hash(
        {
            "identity_kind": "execution",
            "identity": identity.model_dump(mode="json"),
        }
    )


def assert_xp006_release_profile(*, transformers_version: str, torch_version: str) -> None:
    """Refuse runtimes outside the dependency profile that produced xp-006."""
    if transformers_version != XP006_TRANSFORMERS_VERSION:
        raise ReleaseProfileError(
            "xp-006 requires transformers exactly "
            f"{XP006_TRANSFORMERS_VERSION}; found {transformers_version}"
        )
    torch_major_minor = _major_minor(torch_version)
    if not (
        torch_major_minor is not None
        and XP006_TORCH_MIN <= torch_major_minor < XP006_TORCH_MAX_EXCLUSIVE
    ):
        raise ReleaseProfileError(f"xp-006 requires torch >=2.6,<2.7; found {torch_version}")


def assert_installed_xp006_release_profile() -> dict[str, str]:
    """Validate installed package metadata without importing Torch or Transformers."""
    try:
        versions = {
            "transformers": metadata.version("transformers"),
            "torch": metadata.version("torch"),
        }
    except metadata.PackageNotFoundError as exc:
        raise ReleaseProfileError(
            f"xp-006 release dependency is not installed: {exc.name}"
        ) from exc
    assert_xp006_release_profile(
        transformers_version=versions["transformers"],
        torch_version=versions["torch"],
    )
    return versions


__all__ = [
    "UNKNOWN_EXECUTION_VALUE",
    "XP006_TRANSFORMERS_VERSION",
    "ExecutionIdentity",
    "ReleaseProfileError",
    "assert_installed_xp006_release_profile",
    "assert_xp006_release_profile",
    "execution_fingerprint",
]
