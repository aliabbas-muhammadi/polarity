"""Deterministic, script-free fetch for the cleared sources.

Author-run (this is the only place network access happens; pytest stays offline). Fails closed
on ``LICENSE_UNRESOLVED`` or an unpinned revision, verifies checksums, and extracts only the
named archive members by basename into a flat directory (no path traversal, no code execution,
no ``trust_remote_code``). Raw bytes land in the git-ignored ``data/raw/<id>/``; the per-source
``PROVENANCE.json`` (with per-file sha256) is written to the committed ``data/sources/<id>/``.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import urllib.request
import zipfile
from collections.abc import Callable
from datetime import date
from pathlib import Path

from polarity.data.registry import LicenseStatus
from polarity.data.sources import DatasetSource, DownloadSpec

RAW_ROOT = Path("data/raw")
SOURCE_PROVENANCE_ROOT = Path("data/sources")
_UNPINNED = {"", "main", "master", "head", "latest"}
_MAX_BYTES = 512 * 1024 * 1024  # 512 MB safety bound

Opener = Callable[[str], bytes]


class FetchError(RuntimeError):
    """Raised on a refused, corrupt, or malformed fetch."""


def _looks_like_sha(rev: str) -> bool:
    return len(rev) == 40 and all(c in "0123456789abcdef" for c in rev.lower())


def assert_fetchable(source: DatasetSource) -> None:
    """Fail closed unless the source is cleared AND pinned to immutable bytes."""
    if source.license_status == LicenseStatus.LICENSE_UNRESOLVED:
        raise FetchError(f"{source.id}: LICENSE_UNRESOLVED — refusing to fetch (fail closed)")
    if source.revision.strip().lower() in _UNPINNED:
        raise FetchError(f"{source.id}: revision {source.revision!r} is a floating ref")
    pinned = (
        _looks_like_sha(source.revision)
        or source.revision.startswith("unversioned:")
        or any(d.expected_sha256 for d in source.downloads)
    )
    if not pinned:
        raise FetchError(f"{source.id}: revision {source.revision!r} is not pinned")


def _default_opener(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "polarity-m1/0.1 (+research)"})
    with urllib.request.urlopen(request, timeout=180) as resp:
        data: bytes = resp.read(_MAX_BYTES + 1)
    if len(data) > _MAX_BYTES:
        raise FetchError(f"{url}: exceeds size bound {_MAX_BYTES} bytes")
    return data


def _matching_members(names: list[str], wanted: set[str]) -> dict[str, str]:
    """Map each wanted basename to its actual (possibly nested) archive member name."""
    found: dict[str, str] = {}
    for name in names:
        base = name.rsplit("/", 1)[-1]
        if base in wanted and base not in found:
            found[base] = name
    return found


def _extract(data: bytes, kind: str, members: tuple[str, ...], dest_dir: Path) -> dict[str, Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path] = {}
    if kind == "tar.gz":
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            found = _matching_members(tf.getnames(), set(members))
            for base, member in found.items():
                info = tf.getmember(member)
                extracted = tf.extractfile(member) if info.isfile() else None
                if extracted is None:
                    continue
                out = dest_dir / base  # flat write — never the archive's own path
                out.write_bytes(extracted.read(_MAX_BYTES + 1))
                results[base] = out
    elif kind == "zip":
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            found = _matching_members(zf.namelist(), set(members))
            for base, member in found.items():
                out = dest_dir / base
                out.write_bytes(zf.read(member))
                results[base] = out
    else:
        raise FetchError(f"unknown archive kind: {kind!r}")
    missing = set(members) - set(results)
    if missing:
        raise FetchError(f"{kind}: archive missing expected members {sorted(missing)}")
    return results


def _parquet_to_jsonl(data: bytes, spec: DownloadSpec, dest_dir: Path) -> Path:
    """Convert a downloaded parquet file to JSONL (keeping only ``spec.columns``).

    Keeps the adapters + tests JSONL-based and lossless; parquet is a data-only columnar format
    (no code execution). ``pyarrow`` is imported lazily so it is only needed for fetching.
    """
    import pyarrow.parquet as pq  # lazy: fetch-only dependency

    name = spec.dest or "data.jsonl"
    tmp = dest_dir / f"{name}.parquet.tmp"
    tmp.write_bytes(data)
    try:
        table = pq.read_table(tmp, columns=list(spec.columns) or None)
    finally:
        tmp.unlink(missing_ok=True)
    out = dest_dir / name
    with out.open("w", encoding="utf-8") as handle:
        for row in table.to_pylist():
            handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=True) + "\n")
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source_provenance(
    source: DatasetSource, file_shas: dict[str, str], provenance_root: Path
) -> Path:
    out_dir = provenance_root / source.id
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": source.id,
        "name": source.name,
        "homepage": source.homepage,
        "tier": source.tier.value,
        "revision": source.revision,
        "format": source.format.value,
        "license": source.license,
        "license_status": source.license_status.value,
        "license_url": source.license_url,
        "attribution": source.attribution,
        "retrieved": date.today().isoformat(),
        "files": dict(sorted(file_shas.items())),
    }
    path = out_dir / "PROVENANCE.json"
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path


def fetch(
    source: DatasetSource,
    *,
    raw_root: Path = RAW_ROOT,
    provenance_root: Path = SOURCE_PROVENANCE_ROOT,
    opener: Opener = _default_opener,
) -> Path:
    """Download + verify + extract a source into ``raw_root/<id>/``; write its provenance."""
    assert_fetchable(source)
    dest_dir = raw_root / source.id
    file_shas: dict[str, str] = {}
    for spec in source.downloads:
        data = opener(spec.url)
        got = hashlib.sha256(data).hexdigest()
        if got != spec.expected_sha256:
            raise FetchError(
                f"{source.id}: sha256 mismatch for {spec.url}: {got} != {spec.expected_sha256}"
            )
        dest_dir.mkdir(parents=True, exist_ok=True)
        if spec.kind == "raw":
            name = spec.dest or spec.url.rsplit("/", 1)[-1]
            out = dest_dir / name
            out.write_bytes(data)
            file_shas[name] = _sha256(out)
        elif spec.kind == "parquet":
            out = _parquet_to_jsonl(data, spec, dest_dir)
            file_shas[out.name] = _sha256(out)
        else:
            for base, path in _extract(data, spec.kind, spec.members, dest_dir).items():
                file_shas[base] = _sha256(path)
    _write_source_provenance(source, file_shas, provenance_root)
    return dest_dir


__all__ = ["RAW_ROOT", "SOURCE_PROVENANCE_ROOT", "FetchError", "assert_fetchable", "fetch"]
