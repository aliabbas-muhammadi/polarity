"""CANNOT adapter: TSV `premise, hypothesis, label` (0=non-negated, 1=negated).

The negation slice: 0 (non-negated paraphrase) -> CACHE_EQUIVALENT, 1 (negated) ->
NOT_CACHE_EQUIVALENT. Both classes from one source.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from polarity.data.mapping import MappingAudit, make_example
from polarity.data.schema import Example, RecipeTier
from polarity.data.sources import DatasetSource
from polarity.types import Label

_FILE = "cannot_dataset_v1.1.tsv"


def adapt(source: DatasetSource, raw_dir: Path) -> tuple[list[Example], MappingAudit]:
    examples: list[Example] = []
    rejections: Counter[str] = Counter()
    source_rows = 0

    path = raw_dir / _FILE
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for i, raw_line in enumerate(handle):
                line = raw_line.rstrip("\n")
                if not line.strip():
                    continue
                parts = line.split("\t")
                if i == 0 and len(parts) == 3 and parts[2].strip().lower() not in {"0", "1"}:
                    continue  # header row
                source_rows += 1
                if len(parts) != 3:
                    rejections["malformed_row"] += 1
                    continue
                premise, hypothesis, lab = parts[0], parts[1], parts[2].strip()
                if lab == "0":
                    label, rule = Label.CACHE_EQUIVALENT, "cannot:label0(non-negated)->equivalent"
                elif lab == "1":
                    label, rule = (
                        Label.NOT_CACHE_EQUIVALENT,
                        "cannot:label1(negated)->not_equivalent",
                    )
                else:
                    rejections["label_not_binary"] += 1
                    continue
                if not premise.strip() or not hypothesis.strip():
                    rejections["empty_text"] += 1
                    continue
                examples.append(
                    make_example(
                        source=source.id,
                        revision=source.revision,
                        source_split="all",
                        source_example_id=str(i),
                        text_a=premise,
                        text_b=hypothesis,
                        original_label=lab,
                        label=label,
                        mapping_rule=rule,
                        license_id=source.id,
                        tier=RecipeTier.PRIMARY,
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
