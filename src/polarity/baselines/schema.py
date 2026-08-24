"""M2 prediction record: the M0 Prediction fields + a few shortcut-analysis features.

Features are computed at inference time (where text is available) and committed alongside the
score, so the keyless report can do shortcut analysis without reconstructing text.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from polarity.data.categorize import NEGATION_CUES
from polarity.textnorm import normalize
from polarity.types import Category, Label, Prediction

_TOKEN = re.compile(r"[a-z0-9']+")


class BaselinePrediction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    pair_id: str
    gold: Label
    score: float = Field(ge=0.0, le=1.0)
    category: Category
    source: str
    has_neg_cue: bool  # either text carries an explicit negation cue
    token_overlap: float = Field(ge=0.0, le=1.0)  # Jaccard over normalized tokens
    len_diff: int  # |token-count(a) - token-count(b)|

    def core(self) -> Prediction:
        return Prediction(
            pair_id=self.pair_id,
            gold=self.gold,
            score=self.score,
            category=self.category,
            source=self.source,
        )


def align_predictions_exact(
    left: list[BaselinePrediction],
    right: list[BaselinePrediction],
    *,
    left_name: str,
    right_name: str,
) -> list[tuple[BaselinePrediction, BaselinePrediction]]:
    """Return paired rows only when ordered IDs and comparison metadata match exactly."""
    if len(left) != len(right):
        raise ValueError(
            f"paired prediction count mismatch: {left_name}={len(left)}, {right_name}={len(right)}"
        )

    aligned: list[tuple[BaselinePrediction, BaselinePrediction]] = []
    for index, (left_pred, right_pred) in enumerate(zip(left, right, strict=True)):
        if left_pred.pair_id != right_pred.pair_id:
            raise ValueError(
                f"paired pair_id mismatch at position {index}: "
                f"{left_name}={left_pred.pair_id!r}, {right_name}={right_pred.pair_id!r}"
            )
        if left_pred.gold != right_pred.gold:
            raise ValueError(
                f"paired gold label mismatch for pair_id {left_pred.pair_id}: "
                f"{left_name}={left_pred.gold.value!r}, {right_name}={right_pred.gold.value!r}"
            )
        if left_pred.category != right_pred.category:
            raise ValueError(
                f"paired category mismatch for pair_id {left_pred.pair_id}: "
                f"{left_name}={left_pred.category.value!r}, "
                f"{right_name}={right_pred.category.value!r}"
            )
        if left_pred.source != right_pred.source:
            raise ValueError(
                f"paired source mismatch for pair_id {left_pred.pair_id}: "
                f"{left_name}={left_pred.source!r}, {right_name}={right_pred.source!r}"
            )
        aligned.append((left_pred, right_pred))
    return aligned


def shortcut_features(text_a: str, text_b: str) -> tuple[bool, float, int]:
    ta = set(_TOKEN.findall(normalize(text_a)))
    tb = set(_TOKEN.findall(normalize(text_b)))
    union = ta | tb
    overlap = len(ta & tb) / len(union) if union else 0.0
    has_cue = bool((NEGATION_CUES & ta) or (NEGATION_CUES & tb))
    len_diff = abs(len(text_a.split()) - len(text_b.split()))
    return has_cue, round(overlap, 6), len_diff


__all__ = ["BaselinePrediction", "align_predictions_exact", "shortcut_features"]
