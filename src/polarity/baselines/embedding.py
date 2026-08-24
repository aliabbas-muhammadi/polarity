"""Frozen sentence-embedding cosine baseline. Lazy-imports sentence-transformers.

Score = (cosine + 1) / 2 in [0, 1], symmetric by construction. The model's own pooling +
normalization are applied by SentenceTransformer; bge-small needs no instruction prefix for
symmetric similarity (registry ``prefix`` is empty).
"""

from __future__ import annotations

from typing import Any

from polarity.baselines.registry import ModelSpec


class EmbeddingScorer:
    def __init__(self, spec: ModelSpec) -> None:
        from sentence_transformers import SentenceTransformer

        self.spec = spec
        self._model = SentenceTransformer(
            spec.hf_id,
            revision=spec.revision,
            device="cpu",
            trust_remote_code=spec.trust_remote_code,
        )

    def encode(self, texts: list[str]) -> Any:
        prepared = [self.spec.prefix + t for t in texts] if self.spec.prefix else texts
        return self._model.encode(
            prepared,
            normalize_embeddings=self.spec.normalize,
            convert_to_numpy=True,
            batch_size=64,
        )

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        import numpy as np

        unique = sorted({t for pair in pairs for t in pair})
        row = {t: i for i, t in enumerate(unique)}
        emb = self.encode(unique)
        out: list[float] = []
        for text_a, text_b in pairs:
            va, vb = emb[row[text_a]], emb[row[text_b]]
            denom = float(np.linalg.norm(va)) * float(np.linalg.norm(vb))
            cos = float(np.dot(va, vb)) / denom if denom else 0.0
            out.append(max(0.0, min(1.0, (cos + 1.0) / 2.0)))
        return out


__all__ = ["EmbeddingScorer"]
