"""NaN-NLI adapter (EVAL PROBE): sub-clausal negation test suite, never a training source.

Kept as an isolated eval probe. The upstream 3-way NLI label is retained in ``original_label``
but NOT mapped to a cache-equivalence label (``label = None``) — a probe is measured against a
frozen model later, not trained on. All rows are kept.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from polarity.data.mapping import MappingAudit, make_example
from polarity.data.schema import Example, RecipeTier
from polarity.data.sources import DatasetSource

_FILE = "nan.csv"


def adapt(source: DatasetSource, raw_dir: Path) -> tuple[list[Example], MappingAudit]:
    examples: list[Example] = []
    rejections: Counter[str] = Counter()
    source_rows = 0

    path = raw_dir / _FILE
    if path.exists():
        with path.open(encoding="utf-8", newline="") as handle:
            for i, row in enumerate(csv.DictReader(handle)):
                source_rows += 1
                premise = (row.get("premise") or "").strip()
                hypothesis = (row.get("hypothesis") or "").strip()
                gold = (row.get("label") or "").strip()
                if not premise or not hypothesis:
                    rejections["empty_text"] += 1
                    continue
                examples.append(
                    make_example(
                        source=source.id,
                        revision=source.revision,
                        source_split="probe",
                        source_example_id=str(i),
                        text_a=premise,
                        text_b=hypothesis,
                        original_label=gold,
                        label=None,  # eval probe: never a training label
                        mapping_rule="nan-nli:eval-probe-unmapped",
                        license_id=source.id,
                        tier=RecipeTier.EVAL_PROBE,
                    )
                )

    audit = MappingAudit(
        source=source.id,
        source_rows=source_rows,
        accepted=len(examples),
        rejected=sum(rejections.values()),
        rejections=dict(rejections),
    )
    return examples, audit
