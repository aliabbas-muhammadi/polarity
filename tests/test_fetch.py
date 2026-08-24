import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from polarity.data.fetch import FetchError, assert_fetchable, fetch
from polarity.data.registry import LicenseStatus
from polarity.data.schema import RecipeTier
from polarity.data.sources import SOURCES, DatasetSource, DownloadSpec, FileFormat


def _targz(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _src(**kw: object) -> DatasetSource:
    base: dict[str, object] = {
        "id": "t",
        "name": "t",
        "homepage": "h",
        "tier": RecipeTier.PRIMARY,
        "revision": "unversioned:test",
        "format": FileFormat.TSV,
        "license": "x",
        "license_status": LicenseStatus.USABLE,
        "license_url": "u",
        "attribution": "a",
        "downloads": (),
    }
    base.update(kw)
    return DatasetSource(**base)  # type: ignore[arg-type]


def test_registry_sources_are_fetchable():
    for src in SOURCES.values():
        assert_fetchable(src)


def test_license_unresolved_refused():
    src = SOURCES["cannot"].model_copy(update={"license_status": LicenseStatus.LICENSE_UNRESOLVED})
    with pytest.raises(FetchError):
        assert_fetchable(src)


def test_floating_revision_refused():
    src = SOURCES["cannot"].model_copy(
        update={
            "revision": "main",
            "downloads": (
                DownloadSpec(url="http://x/y", kind="raw", dest="y", expected_sha256="0" * 64),
            ),
        }
    )
    with pytest.raises(FetchError):
        assert_fetchable(src)


def test_fetch_extracts_named_members(tmp_path: Path):
    payload = _targz({"final/train.tsv": b"id\ts1\ts2\tlabel\n", "final/other.txt": b"ignore"})
    src = _src(
        downloads=(
            DownloadSpec(
                url="http://x/d.tar.gz",
                kind="tar.gz",
                expected_sha256=hashlib.sha256(payload).hexdigest(),
                members=("train.tsv",),
            ),
        )
    )
    fetch(
        src,
        raw_root=tmp_path / "raw",
        provenance_root=tmp_path / "sources",
        opener=lambda _url: payload,
    )
    assert (tmp_path / "raw" / "t" / "train.tsv").exists()
    assert not (tmp_path / "raw" / "t" / "other.txt").exists()
    assert (tmp_path / "sources" / "t" / "PROVENANCE.json").exists()


def test_fetch_sha256_mismatch_fails_before_writing(tmp_path: Path):
    spec = DownloadSpec(
        url="http://x/f.tsv",
        kind="raw",
        dest="f.tsv",
        expected_sha256="0534942de772c5ee51ef5177729e073819f2b1e39458870f18253a4bb3bb4f06",
    )
    src = _src(downloads=(spec,))

    with pytest.raises(FetchError, match="sha256 mismatch"):
        fetch(
            src,
            raw_root=tmp_path / "raw",
            provenance_root=tmp_path / "sources",
            opener=lambda _url: b"tampered",
        )

    assert not (tmp_path / "raw" / "t").exists()
    assert not (tmp_path / "sources" / "t" / "PROVENANCE.json").exists()
