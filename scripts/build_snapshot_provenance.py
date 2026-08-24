"""Deterministically (re)build the sealed snapshot's PROVENANCE.json.

Run: ``uv run python scripts/build_snapshot_provenance.py``

The snapshot ``golden.json`` is a verbatim copy of the sibling veritas repo's
``eval/golden.json`` (pair text + labels only). This records its checksum and the sealing
governance so ``polarity data verify`` and CI can gate it. Re-run only if the snapshot is
intentionally re-imported.
"""

from __future__ import annotations

from pathlib import Path

from polarity.data.loaders import load_golden
from polarity.data.registry import DatasetRecord, LicenseStatus, write_provenance
from polarity.fingerprint import sha256_file

SNAP = Path("data/snapshots/veritas-golden")
GOLDEN = SNAP / "golden.json"


def main() -> None:
    golden = load_golden(GOLDEN)
    record = DatasetRecord(
        name="veritas-golden",
        role="final_holdout",
        source="veritas (sibling repo, same author) :: eval/golden.json",
        revision="164dc8e734b52732578c4aa44a872e674900ae1d",
        license="author-owned internal evaluation set (see NOTICE)",
        license_status=LicenseStatus.USABLE,
        retrieved="2026-08-20",
        rows=len(golden),
        sha256=sha256_file(GOLDEN),
        changes=(
            "verbatim copy; pair text + labels only "
            "(id, group, base, probe, pairType, equivalent, note); "
            "NO embeddings, NO guard verdicts imported"
        ),
        sealed=True,
        notes=(
            "SEALED FINAL HOLDOUT (governance G1/G2, docs/METHODOLOGY.md): never used for "
            "training, threshold selection, calibration, augmentation, or model/checkpoint "
            "selection; reserved for one final transfer evaluation at M6. "
            "veritas HEAD at capture: bd36420b18ccaa58a535866a0ff20da56c83fac7."
        ),
    )
    write_provenance(SNAP / "PROVENANCE.json", record)
    print(f"wrote {SNAP / 'PROVENANCE.json'} (rows={record.rows}, sha256={record.sha256[:12]})")


if __name__ == "__main__":
    main()
