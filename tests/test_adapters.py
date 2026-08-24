import json
from pathlib import Path

from polarity.data.adapters import cannot, mnli, nan_nli, paws_wiki
from polarity.data.schema import RecipeTier
from polarity.data.sources import get_source
from polarity.types import Label


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_paws_adapter(tmp_path: Path):
    rows = [
        {"id": 1, "sentence1": "Cat sat", "sentence2": "The cat sat", "label": 1},
        {"id": 2, "sentence1": "A to B", "sentence2": "B to A", "label": 0},
        {"id": 3, "sentence1": "", "sentence2": "x", "label": 1},  # empty text -> rejected
    ]
    _write(tmp_path / "train.jsonl", "\n".join(json.dumps(r) for r in rows) + "\n")
    examples, audit = paws_wiki.adapt(get_source("paws-wiki"), tmp_path)
    assert (audit.source_rows, audit.accepted, audit.rejected) == (3, 2, 1)
    assert audit.reconciles()
    labels = {e.source_example_id: e.label for e in examples}
    assert labels["1"] == Label.CACHE_EQUIVALENT
    assert labels["2"] == Label.NOT_CACHE_EQUIVALENT


def test_cannot_adapter(tmp_path: Path):
    _write(
        tmp_path / "cannot_dataset_v1.1.tsv",
        "premise\thypothesis\tlabel\nX is good\tX is good indeed\t0\nX is safe\tX is unsafe\t1\n",
    )
    examples, audit = cannot.adapt(get_source("cannot"), tmp_path)
    assert audit.accepted == 2
    by_text = {e.text_a: e.label for e in examples}
    assert by_text["X is good"] == Label.CACHE_EQUIVALENT
    assert by_text["X is safe"] == Label.NOT_CACHE_EQUIVALENT


def test_mnli_adapter_keeps_only_contradiction(tmp_path: Path):
    # HuggingFace integer labels: 0=entailment, 1=neutral, 2=contradiction, -1=no-consensus
    rows = [
        json.dumps({"pairID": f"p{i}", "premise": "s1", "hypothesis": "s2", "label": lab})
        for i, lab in enumerate([2, 0, 1, -1])
    ]
    _write(tmp_path / "train.jsonl", "\n".join(rows) + "\n")
    examples, audit = mnli.adapt(get_source("mnli"), tmp_path)
    assert audit.source_rows == 4
    assert audit.accepted == 1
    assert examples[0].label == Label.NOT_CACHE_EQUIVALENT
    assert audit.rejections["entailment_directional_unsuitable"] == 1
    assert audit.rejections["neutral_ambiguous"] == 1
    assert audit.rejections["no_consensus"] == 1


def test_nan_nli_adapter_is_unmapped_eval_probe(tmp_path: Path):
    _write(tmp_path / "nan.csv", "premise,hypothesis,label\nA,B,contradiction\nC,D,entailment\n")
    examples, audit = nan_nli.adapt(get_source("nan-nli"), tmp_path)
    assert audit.accepted == 2
    assert all(e.label is None for e in examples)
    assert all(e.recipe_tier == RecipeTier.EVAL_PROBE for e in examples)
