"""M5 diagnostic tests (brief §27). Torch-free / offline — the version guard, the trivial-batch
generator, the label convention, and the overfit verdict. The overfit run itself is author-run
(needs the ``train`` extra + a model download), so it is exercised by `polarity diagnose overfit`,
not in CI.
"""

import pytest

from polarity.diagnose import (
    OverfitResult,
    _verdict,
    transformers_supported,
    trivial_pairs,
)
from polarity.train.predict import label_index
from polarity.types import Label


@pytest.mark.parametrize(
    ("version", "supported"),
    [
        ("4.44.0", True),
        ("4.46.3", True),  # tested good (M5)
        ("4.57.6", True),  # what the pinned `train` extra resolves to
        ("4.99.9", True),
        ("4.43.9", False),  # below floor
        ("4.40.0", False),
        ("5.0.0", False),  # regression starts at the 5.x major
        ("5.15.1", False),  # the version that collapsed M3/M3.1/M3.2
        ("6.0.0", False),
    ],
)
def test_transformers_version_guard(version: str, supported: bool):
    assert transformers_supported(version) is supported


def test_label_convention_is_stable_and_binary():
    # H1/H2: the training label integer must match the positive-class convention exactly.
    assert label_index(Label.CACHE_EQUIVALENT) == 1
    assert label_index(Label.NOT_CACHE_EQUIVALENT) == 0


def test_trivial_pairs_are_balanced_distinct_and_correctly_labelled():
    pairs = trivial_pairs(8)
    assert len(pairs) == 16
    pos = [p for p in pairs if p.gold == Label.CACHE_EQUIVALENT]
    neg = [p for p in pairs if p.gold == Label.NOT_CACHE_EQUIVALENT]
    assert len(pos) == len(neg) == 8
    assert len({p.example_id for p in pairs}) == 16  # distinct ids
    # positives: text_a == text_b (mechanically equivalent); negatives: different
    assert all(p.text_a == p.text_b for p in pos)
    assert all(p.text_a != p.text_b for p in neg)


def test_verdict_requires_accuracy_low_loss_and_separation():
    assert _verdict(1.0, 0.01, 0.99) == "OVERFIT_PASS"
    assert _verdict(0.5, 0.70, 0.00) == "OVERFIT_FAIL"  # the collapse signature
    assert _verdict(1.0, 0.50, 0.99) == "OVERFIT_FAIL"  # accurate but loss too high
    assert _verdict(1.0, 0.01, 0.20) == "OVERFIT_FAIL"  # no real separation
    assert _verdict(0.94, 0.01, 0.99) == "OVERFIT_FAIL"  # just under the accuracy bar


def test_overfit_result_as_dict_is_complete():
    r = OverfitResult(
        transformers_version="4.57.6",
        device="cpu",
        n_pairs=16,
        steps=150,
        lr=1e-5,
        prior_ce=0.693,
        final_loss=0.006,
        final_accuracy=1.0,
        prob_min=0.003,
        prob_max=0.998,
        prob_spread=0.995,
        pooled_repr_std=0.412,
        max_grad_norm=3.25,
        delta_head_norm=0.015,
        delta_backbone_norm=0.064,
        verdict="OVERFIT_PASS",
    )
    d = r.as_dict()
    assert d["verdict"] == "OVERFIT_PASS" and d["transformers_version"] == "4.57.6"
    assert d["history"] == []
