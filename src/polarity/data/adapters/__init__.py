"""Per-source adapters: parse raw files into canonical Examples with explicit label mapping.

Each ``adapt(source, raw_dir)`` returns ``(examples, MappingAudit)`` and never downloads —
raw files must already exist under ``raw_dir`` (fetched separately). Kept deliberately small
and pure so they can be unit-tested on tiny synthetic fixtures with no network.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from polarity.data.adapters import cannot, mnli, nan_nli, paws_wiki
from polarity.data.mapping import MappingAudit
from polarity.data.schema import Example
from polarity.data.sources import DatasetSource

Adapter = Callable[[DatasetSource, Path], tuple[list[Example], MappingAudit]]

ADAPTERS: dict[str, Adapter] = {
    "paws-wiki": paws_wiki.adapt,
    "cannot": cannot.adapt,
    "mnli": mnli.adapt,
    "nan-nli": nan_nli.adapt,
}


def adapt(source: DatasetSource, raw_dir: Path) -> tuple[list[Example], MappingAudit]:
    if source.id not in ADAPTERS:
        raise KeyError(f"no adapter for source {source.id!r}")
    return ADAPTERS[source.id](source, raw_dir)


__all__ = ["ADAPTERS", "Adapter", "adapt"]
