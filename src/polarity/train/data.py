"""Tokenization + batching + the token-length measurement. Lazy-imports transformers/torch.

Pair encoding is ``tokenizer(a, b)`` -> ``[CLS] a [SEP] b [SEP]`` (do not hand-insert specials);
DeBERTa-v3 has ``type_vocab_size=0`` so token_type_ids are ignored. Truncation is ``longest_first``
(symmetric task — neither query privileged). ``length_stats`` measures the JOINT pair-token
distribution so ``max_length`` is chosen from evidence, not assumed (brief §5).
"""

from __future__ import annotations

from typing import Any

from polarity.baselines.materialize import MaterializedPair
from polarity.train.config import TrainConfig
from polarity.train.predict import label_index


def _percentile(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, int(q * (len(sorted_vals) - 1) + 0.5))
    return sorted_vals[idx]


def joint_token_lengths(pairs: list[MaterializedPair], tokenizer: Any) -> list[int]:
    """Length of `[CLS] a [SEP] b [SEP]` per pair, WITHOUT truncation (for the max_length study)."""
    lengths: list[int] = []
    for p in pairs:
        enc = tokenizer(p.text_a, p.text_b, truncation=False, add_special_tokens=True)
        lengths.append(len(enc["input_ids"]))
    return lengths


def length_stats(
    pairs: list[MaterializedPair],
    tokenizer: Any,
    candidates: tuple[int, ...] = (128, 192, 256, 384, 512),
) -> dict[str, Any]:
    lengths = sorted(joint_token_lengths(pairs, tokenizer))
    n = len(lengths)
    return {
        "n_pairs": n,
        "p50": _percentile(lengths, 0.50),
        "p90": _percentile(lengths, 0.90),
        "p95": _percentile(lengths, 0.95),
        "p99": _percentile(lengths, 0.99),
        "max": lengths[-1] if lengths else 0,
        "truncated_fraction_at": {
            str(c): round(sum(1 for length in lengths if length > c) / n, 6) if n else 0.0
            for c in candidates
        },
    }


def encode(tokenizer: Any, pairs: list[MaterializedPair], cfg: TrainConfig) -> Any:
    return tokenizer(
        [p.text_a for p in pairs],
        [p.text_b for p in pairs],
        truncation=cfg.training.truncation,
        max_length=cfg.training.max_length,
        padding=True,
        return_tensors="pt",
    )


def labels_tensor(pairs: list[MaterializedPair]) -> Any:
    import torch

    return torch.tensor([label_index(p.gold) for p in pairs], dtype=torch.long)


__all__ = ["encode", "joint_token_lengths", "labels_tensor", "length_stats"]
