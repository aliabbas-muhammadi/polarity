"""Off-the-shelf NLI cross-encoder baseline. Lazy-imports transformers/torch.

Cache-equivalence is symmetric but entailment is directional, so the score is the STRICT
bidirectional combiner ``min(P_entail(a,b), P_entail(b,a))``. The entailment index is read from
``config.label2id`` (case-insensitively) — never hardcoded, because it differs across models.
"""

from __future__ import annotations

from polarity.baselines.registry import ModelSpec


def symmetric_min(forward: list[float], backward: list[float]) -> list[float]:
    """Strict symmetric combiner: cache-equivalence needs entailment in BOTH directions."""
    return [min(x, y) for x, y in zip(forward, backward, strict=True)]


class NliScorer:
    def __init__(self, spec: ModelSpec) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.spec = spec
        self._tok = AutoTokenizer.from_pretrained(spec.hf_id, revision=spec.revision)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            spec.hf_id, revision=spec.revision, trust_remote_code=spec.trust_remote_code
        )
        self._model.eval()
        label2id = {str(k).lower(): int(v) for k, v in self._model.config.label2id.items()}
        if "entailment" not in label2id:
            raise ValueError(f"model exposes no 'entailment' label: {sorted(label2id)}")
        self._entail = label2id["entailment"]

    def _p_entail(self, premises: list[str], hypotheses: list[str]) -> list[float]:
        import torch

        enc = self._tok(
            premises, hypotheses, truncation=True, max_length=512, padding=True, return_tensors="pt"
        )
        with torch.no_grad():
            probs = self._model(**enc).logits.softmax(dim=-1)
        return [float(p) for p in probs[:, self._entail].tolist()]

    def _batched(self, premises: list[str], hypotheses: list[str], batch_size: int) -> list[float]:
        out: list[float] = []
        for i in range(0, len(premises), batch_size):
            out.extend(self._p_entail(premises[i : i + batch_size], hypotheses[i : i + batch_size]))
        return out

    def score(self, pairs: list[tuple[str, str]], *, batch_size: int = 32) -> list[float]:
        text_a = [p[0] for p in pairs]
        text_b = [p[1] for p in pairs]
        forward = self._batched(text_a, text_b, batch_size)
        backward = self._batched(text_b, text_a, batch_size)
        return symmetric_min(forward, backward)


__all__ = ["NliScorer", "symmetric_min"]
