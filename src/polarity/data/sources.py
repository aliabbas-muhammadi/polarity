"""Declarative registry of the license-cleared M1 sources.

The single source of truth for what M1 may download, at exactly which pinned revision, under
which license. ``license_status`` gates fetching: anything not explicitly cleared fails closed
(see fetch.py). Revisions are content-addressable (GitHub raw @ commit SHA, or an archive
pinned by checksum). No source here carries a loader script, so no code execution is triggered.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from polarity.data.registry import LicenseStatus
from polarity.data.schema import RecipeTier


class FileFormat(StrEnum):
    TSV = "tsv"
    JSONL = "jsonl"
    CSV = "csv"


class DownloadSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str
    kind: str  # "raw" | "tar.gz" | "zip" | "parquet"
    expected_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dest: str | None = None  # raw/parquet: local output filename
    members: tuple[str, ...] = ()  # archive: basenames to extract
    columns: tuple[str, ...] = ()  # parquet: columns to keep, written to a JSONL `dest`


class DatasetSource(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    name: str
    homepage: str
    tier: RecipeTier
    revision: str  # pinned commit SHA, or "unversioned:<note>" (then pinned by checksum)
    format: FileFormat
    license: str
    license_status: LicenseStatus
    license_url: str
    attribution: str  # one line for NOTICE / ATTRIBUTIONS.md
    downloads: tuple[DownloadSpec, ...]


SOURCES: dict[str, DatasetSource] = {
    "paws-wiki": DatasetSource(
        id="paws-wiki",
        name="PAWS-Wiki (labeled_final)",
        homepage="https://github.com/google-research-datasets/paws",
        tier=RecipeTier.PRIMARY,
        # The original GCS tarball now returns HTTP 403; the HuggingFace parquet mirror is the
        # maintained, content-addressable source (pinned by the commit SHA below).
        revision="161ece9501cf0a11f3e48bd356eaa82de46d6a09",
        format=FileFormat.JSONL,
        license="Google permissive notice; underlying Wikipedia text CC-BY-SA-3.0",
        license_status=LicenseStatus.USABLE_WITH_CONDITIONS,
        license_url="https://github.com/google-research-datasets/paws/blob/master/LICENSE",
        attribution="PAWS-Wiki (Zhang, Baldridge, He, NAACL 2019), Google Research; "
        "sentences derived from Wikipedia (CC-BY-SA-3.0). Attribution + share-alike apply.",
        downloads=tuple(
            DownloadSpec(
                url="https://huggingface.co/datasets/google-research-datasets/paws/resolve/"
                f"161ece9501cf0a11f3e48bd356eaa82de46d6a09/labeled_final/{hf}-00000-of-00001.parquet",
                kind="parquet",
                expected_sha256={
                    "train": "8dc9ad3e5f30ad9a86b290fe236d528ef23a5751fec9a35d99cbacf68ba277cf",
                    "validation": (
                        "7760d829453764ba342a6f562809a8ed21c2c3eec3fd9ffa544089f145d42f6d"
                    ),
                    "test": "ae342ff12bb84b84b95f468abf5db6cb7c7bd578271299fe9c99be75b8132f4d",
                }[hf],
                dest=f"{local}.jsonl",
                columns=("id", "sentence1", "sentence2", "label"),
            )
            for hf, local in (("train", "train"), ("validation", "validation"), ("test", "test"))
        ),
    ),
    "cannot": DatasetSource(
        id="cannot",
        name="CANNOT (v1.1)",
        homepage="https://github.com/dmlls/cannot-dataset",
        tier=RecipeTier.PRIMARY,
        revision="3b94fb51fcf8e642ea28cdc065805d1d3da3247f",
        format=FileFormat.TSV,
        license="CC-BY-SA-4.0",
        license_status=LicenseStatus.USABLE_WITH_CONDITIONS,
        license_url="https://github.com/dmlls/cannot-dataset#readme",
        attribution="CANNOT dataset (Anschütz, Lozano, Groh, 2023), CC-BY-SA-4.0. "
        "Attribution + share-alike apply.",
        downloads=(
            DownloadSpec(
                url="https://raw.githubusercontent.com/dmlls/cannot-dataset/"
                "3b94fb51fcf8e642ea28cdc065805d1d3da3247f/cannot-dataset/cannot_dataset_v1.1.tsv",
                kind="raw",
                expected_sha256="e3d1c77ed05d4fc2174b7d6d784c5d0b80ec1412d6a18cb721a96a63380d533f",
                dest="cannot_dataset_v1.1.tsv",
            ),
        ),
    ),
    "mnli": DatasetSource(
        id="mnli",
        name="MultiNLI (MNLI)",
        homepage="https://huggingface.co/datasets/nyu-mll/multi_nli",
        tier=RecipeTier.EXPLORATORY_MNLI,
        # HuggingFace parquet mirror (fast CDN, content-addressable SHA) rather than the slow,
        # un-versioned NYU zip.
        revision="da70db2af9d09693783c3320c4249840212ee221",
        format=FileFormat.JSONL,
        license="OANC + CC-BY-3.0 / CC-BY-SA-3.0 / US-public-domain (multi-license)",
        license_status=LicenseStatus.USABLE_WITH_CONDITIONS,
        license_url="https://huggingface.co/datasets/nyu-mll/multi_nli",
        attribution="MultiNLI (Williams, Nangia, Bowman, NAACL 2018). "
        "Multi-license; attribution + share-alike on CC-BY-SA-derived text.",
        downloads=tuple(
            DownloadSpec(
                url="https://huggingface.co/datasets/nyu-mll/multi_nli/resolve/"
                f"da70db2af9d09693783c3320c4249840212ee221/data/{hf}-00000-of-00001.parquet",
                kind="parquet",
                expected_sha256={
                    "train": "1c1de03640b168e410aabfca19e7cc2f3dfcd7f0e126e935674e56fb102c4529",
                    "validation_matched": (
                        "350c26950b55f460b50d36c76aef87d64b49c78812d7abf7bf97e5fede10f186"
                    ),
                    "validation_mismatched": (
                        "6b0e1231cebedd255000a7d1732af37ad0500902239db625e231ed77c0c8f2f8"
                    ),
                }[hf],
                dest=f"{local}.jsonl",
                columns=("pairID", "premise", "hypothesis", "label"),
            )
            for hf, local in (
                ("train", "train"),
                ("validation_matched", "dev_matched"),
                ("validation_mismatched", "dev_mismatched"),
            )
        ),
    ),
    "nan-nli": DatasetSource(
        id="nan-nli",
        name="NaN-NLI (sub-clausal negation test suite)",
        homepage="https://github.com/joey234/nan-nli",
        tier=RecipeTier.EVAL_PROBE,
        revision="1f84029a53940b675b62ab2b60980d4cae26eb78",
        format=FileFormat.CSV,
        license="CC-BY-SA-4.0",
        license_status=LicenseStatus.USABLE_WITH_CONDITIONS,
        license_url="https://huggingface.co/datasets/joey234/nan-nli",
        attribution="NaN-NLI (Truong et al., AACL-IJCNLP 2022), CC-BY-SA-4.0. Eval probe only.",
        downloads=(
            DownloadSpec(
                url="https://raw.githubusercontent.com/joey234/nan-nli/"
                "1f84029a53940b675b62ab2b60980d4cae26eb78/data/nan.csv",
                kind="raw",
                expected_sha256="533cd0a169662834a5bcbb6c064472e52b549424f43164cd7eabc087bfd6e89c",
                dest="nan.csv",
            ),
        ),
    ),
}


def get_source(source_id: str) -> DatasetSource:
    if source_id not in SOURCES:
        raise KeyError(f"unknown source: {source_id!r} (known: {sorted(SOURCES)})")
    return SOURCES[source_id]


__all__ = ["SOURCES", "DatasetSource", "DownloadSpec", "FileFormat", "get_source"]
