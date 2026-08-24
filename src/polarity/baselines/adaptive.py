"""vCache-inspired per-neighborhood adaptive-threshold baseline (offline, keyless).

Faithful offline reduction of vCache (arXiv:2502.03771): vCache fits a per-anchor logistic
P(correct|sim)=expit(gamma*(s-t)). Our source-families are mostly singletons, so per-anchor
observations are too sparse; we fit a **per-neighborhood** logistic over each pair's k nearest
validation anchors and emit the local-sigmoid P(cache-equivalent) as the score. A global sigmoid
is the fallback when a neighborhood is degenerate (< MIN_OBS or single-class), mirroring vCache's
cold-start rule. APPROXIMATION (disclosed in docs/M2-METHODOLOGY.md): the online adaptation, the
stochastic explore/exploit gate, the live LLM judge, and the delta-guarantee validity are NOT
reproduced. Not called "vCache".
"""

from __future__ import annotations

from typing import Any

from polarity.baselines.embedding import EmbeddingScorer
from polarity.baselines.materialize import MaterializedPair
from polarity.baselines.registry import ModelSpec
from polarity.types import Label

DEFAULT_K = 100  # neighborhood size (documented default)
MIN_OBS = 6  # vCache cold-start floor


class AdaptiveScorer:
    def __init__(self, spec: ModelSpec, *, k: int = DEFAULT_K) -> None:
        self._emb = EmbeddingScorer(spec)
        self._k = k
        self._fitted = False
        self._anchor_emb: Any = None
        self._s: Any = None
        self._c: Any = None
        self._global: tuple[float, float] = (0.0, 0.0)

    @staticmethod
    def _unit(mat: Any) -> Any:
        import numpy as np

        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.where(norms == 0, 1.0, norms)

    @staticmethod
    def _fit_logistic(s: Any, c: Any) -> tuple[float, float]:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression()
        model.fit(s.reshape(-1, 1), c)
        return float(model.coef_[0][0]), float(model.intercept_[0])

    def fit(self, validation: list[MaterializedPair]) -> AdaptiveScorer:
        import numpy as np

        anchors = self._unit(self._emb.encode([p.text_a for p in validation]))
        queries = self._unit(self._emb.encode([p.text_b for p in validation]))
        self._anchor_emb = anchors
        self._s = np.array([float(np.dot(anchors[i], queries[i])) for i in range(len(validation))])
        self._c = np.array([1 if p.gold == Label.CACHE_EQUIVALENT else 0 for p in validation])
        self._global = self._fit_logistic(self._s, self._c)
        self._fitted = True
        return self

    def _local_params(self, anchor_vec: Any, exclude: int) -> tuple[float, float]:
        import numpy as np

        sims = self._anchor_emb @ anchor_vec
        order = np.argsort(-sims).tolist()
        neigh = [i for i in order if i != exclude][: self._k]
        c = self._c[neigh]
        if len(neigh) < MIN_OBS or len(set(c.tolist())) < 2:
            return self._global
        return self._fit_logistic(self._s[neigh], c)

    def score(self, pairs: list[MaterializedPair], *, loo: bool) -> list[float]:
        import numpy as np

        if not self._fitted:
            raise RuntimeError("call fit(validation) before score()")
        av = self._unit(self._emb.encode([p.text_a for p in pairs]))
        qv = self._unit(self._emb.encode([p.text_b for p in pairs]))
        out: list[float] = []
        for i in range(len(pairs)):
            s = float(np.dot(av[i], qv[i]))
            gamma, intercept = self._local_params(av[i], i if loo else -1)
            p_equiv = 1.0 / (1.0 + float(np.exp(-(gamma * s + intercept))))
            out.append(max(0.0, min(1.0, p_equiv)))
        return out


__all__ = ["DEFAULT_K", "MIN_OBS", "AdaptiveScorer"]
