"""PAWS-Wiki adapter: JSONL `id, sentence1, sentence2, label` (1=paraphrase, 0=different).

Fetched from the HuggingFace parquet mirror and converted to JSONL by fetch.py. Both classes
come from this one source, which keeps source from determining the label. Upstream
train/validation/test are POOLED here; Polarity re-splits by source-family.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from polarity.data.mapping import MappingAudit, make_example
from polarity.data.schema import Example, RecipeTier
from polarity.data.sources import DatasetSource
from polarity.types import Label

_FILES = {"train.jsonl": "train", "validation.jsonl": "validation", "test.jsonl": "test"}


def adapt(source: DatasetSource, raw_dir: Path) -> tuple[list[Example], MappingAudit]:
    examples: list[Example] = []
    rejections: Counter[str] = Counter()
    source_rows = 0

    for fname, upstream_split in _FILES.items():
        path = raw_dir / fname
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for i, raw_line in enumerate(handle):
                line = raw_line.strip()
                if not line:
                    continue
                source_rows += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    rejections["malformed_json"] += 1
                    continue
                s1, s2, lab = obj.get("sentence1"), obj.get("sentence2"), obj.get("label")
                if (
                    not isinstance(s1, str)
                    or not isinstance(s2, str)
                    or not s1.strip()
                    or not s2.strip()
                ):
                    rejections["empty_text"] += 1
                    continue
                lab_str = str(lab)
                if lab_str == "1":
                    label, rule = Label.CACHE_EQUIVALENT, "paws:label1->equivalent"
                elif lab_str == "0":
                    label, rule = Label.NOT_CACHE_EQUIVALENT, "paws:label0->not_equivalent"
                else:
                    rejections["label_not_binary"] += 1
                    continue
                rid = obj.get("id")
                examples.append(
                    make_example(
                        source=source.id,
                        revision=source.revision,
                        source_split=upstream_split,
                        source_example_id=str(rid) if rid is not None else f"{upstream_split}:{i}",
                        text_a=s1,
                        text_b=s2,
                        original_label=lab_str,
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
