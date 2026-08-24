"""MNLI adapter (EXPLORATORY tier): conservative NLI -> cache-equivalence mapping.

Only ``contradiction`` maps to NOT_CACHE_EQUIVALENT. ``entailment`` is directional (A entails B
does not mean the same answer serves both) and ``neutral`` is ambiguous, so both are REJECTED as
unsuitable rather than forced into a binary label; ``-1`` (no annotator consensus) is dropped.
MNLI therefore contributes negatives only — the source-confound risk the M1 report must quantify.
Upstream train/dev_matched/dev_mismatched are pooled; Polarity re-splits by source-family.

Fetched from the HuggingFace parquet mirror and converted to JSONL by fetch.py. The HF schema
uses an INTEGER label (0=entailment, 1=neutral, 2=contradiction, -1=no-consensus). The train
file is large (~400MB), so it is streamed line by line.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from polarity.data.mapping import MappingAudit, make_example
from polarity.data.schema import Example, RecipeTier
from polarity.data.sources import DatasetSource
from polarity.types import Label

_FILES = {
    "train.jsonl": "train",
    "dev_matched.jsonl": "dev_matched",
    "dev_mismatched.jsonl": "dev_mismatched",
}


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
                premise, hypothesis = obj.get("premise"), obj.get("hypothesis")
                # MNLI pairIDs are NOT unique (some repeat), so the line index guarantees a
                # unique per-file id while retaining the pairID for provenance.
                pid = f"{obj.get('pairID') or upstream_split}-{i}"
                if (
                    not isinstance(premise, str)
                    or not isinstance(hypothesis, str)
                    or not premise.strip()
                    or not hypothesis.strip()
                ):
                    rejections["empty_text"] += 1
                    continue
                lab = obj.get("label")
                if lab == 2:  # contradiction
                    label, rule = Label.NOT_CACHE_EQUIVALENT, "mnli:contradiction->not_equivalent"
                elif lab == 0:  # entailment
                    rejections["entailment_directional_unsuitable"] += 1
                    continue
                elif lab == 1:  # neutral
                    rejections["neutral_ambiguous"] += 1
                    continue
                elif lab == -1:  # no annotator consensus
                    rejections["no_consensus"] += 1
                    continue
                else:
                    rejections["unknown_label"] += 1
                    continue
                examples.append(
                    make_example(
                        source=source.id,
                        revision=source.revision,
                        source_split=upstream_split,
                        source_example_id=pid,
                        text_a=premise,
                        text_b=hypothesis,
                        original_label="contradiction",
                        label=label,
                        mapping_rule=rule,
                        license_id=source.id,
                        tier=RecipeTier.EXPLORATORY_MNLI,
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
