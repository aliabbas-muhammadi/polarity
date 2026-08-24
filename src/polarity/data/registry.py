"""Dataset registry + provenance sidecar.

Every dataset (a public source or the sealed snapshot) gets a ``PROVENANCE.json`` recording
where it came from, at what revision, under what license, and how it was transformed. The
``license_status`` field gates use downstream: anything ``LICENSE_UNRESOLVED`` is excluded
from the M1 build (docs/DATA.md). Provenance is written deterministically so it can be
committed and diffed.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class LicenseStatus(StrEnum):
    USABLE = "USABLE"
    USABLE_WITH_CONDITIONS = "USABLE_WITH_CONDITIONS"
    LICENSE_UNRESOLVED = "LICENSE_UNRESOLVED"


class DatasetRecord(BaseModel):
    """One dataset's provenance + licensing status."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    role: str  # positives | hard_negatives | final_holdout | ...
    source: str
    revision: str  # git sha / HF revision / "n/a"
    license: str
    license_status: LicenseStatus
    retrieved: str  # ISO date
    rows: int
    sha256: str
    changes: str  # transformation notes
    sealed: bool = False
    notes: str = ""


def write_provenance(path: Path, record: DatasetRecord) -> None:
    """Write a deterministic PROVENANCE.json (sorted keys, 2-space indent, trailing NL)."""
    payload = record.model_dump(mode="json")
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def read_provenance(path: Path) -> DatasetRecord:
    return DatasetRecord.model_validate_json(path.read_text(encoding="utf-8"))


__all__ = ["DatasetRecord", "LicenseStatus", "read_provenance", "write_provenance"]
