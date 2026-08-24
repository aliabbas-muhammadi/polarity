"""Fit-on-validation calibrators (temperature / Platt / isotonic), pure and deterministic.

M2's report kept a private grid-search temperature helper; M3 needs a real, *selectable*
calibrator fit on validation and applied frozen to internal_test (brief §15). Everything here is
pure Python float math — no torch, no numpy, no randomness — so a trained model's calibrated
scores reproduce byte-for-byte in the keyless drift gate, independent of library versions.

Convention: inputs are P(CACHE_EQUIVALENT) in [0, 1] and integer labels (1 = positive). Each
fitter returns a ``Calibrator`` with a jsonable ``params`` and an ``apply`` that maps probabilities
to calibrated probabilities. The report fits all candidates on validation, picks the lowest
validation NLL, freezes it, and applies it unchanged to internal_test.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

_EPS = 1e-9


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _clip(p: float) -> float:
    return min(max(p, _EPS), 1.0 - _EPS)


def nll(probs: Sequence[float], labels: Sequence[int]) -> float:
    """Mean negative log-likelihood (lower is better) — the calibration-selection criterion."""
    if len(probs) != len(labels):
        raise ValueError("probs and labels must be the same length")
    if not probs:
        raise ValueError("empty input")
    total = 0.0
    for p, y in zip(probs, labels, strict=True):
        q = _clip(p)
        total -= y * math.log(q) + (1 - y) * math.log(1.0 - q)
    return total / len(probs)


@dataclass(frozen=True)
class Calibrator:
    """A frozen, fit-on-validation calibrator applied unchanged to internal_test."""

    method: str
    params: dict[str, Any] = field(default_factory=dict)
    # isotonic lookup (sorted breakpoints); empty for parametric methods.
    _iso_x: tuple[float, ...] = ()
    _iso_y: tuple[float, ...] = ()

    def apply(self, probs: Sequence[float]) -> list[float]:
        if self.method == "identity":
            return [_clip(float(p)) for p in probs]
        if self.method == "temperature":
            t = float(self.params["temperature"])
            return [_clip(_sigmoid(_logit(p) / t)) for p in probs]
        if self.method == "platt":
            a = float(self.params["a"])
            b = float(self.params["b"])
            return [_clip(_sigmoid(a * _logit(p) + b)) for p in probs]
        if self.method == "isotonic":
            return [_clip(self._iso_lookup(float(p))) for p in probs]
        raise ValueError(f"unknown calibrator method {self.method!r}")

    def _iso_lookup(self, p: float) -> float:
        xs, ys = self._iso_x, self._iso_y
        if not xs:
            return p
        if p <= xs[0]:
            return ys[0]
        if p >= xs[-1]:
            return ys[-1]
        # piecewise-linear interpolation between the fitted breakpoints
        lo, hi = 0, len(xs) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if xs[mid] <= p:
                lo = mid
            else:
                hi = mid
        x0, x1, y0, y1 = xs[lo], xs[hi], ys[lo], ys[hi]
        if x1 == x0:
            return y0
        return y0 + (y1 - y0) * (p - x0) / (x1 - x0)


def identity() -> Calibrator:
    return Calibrator(method="identity")


def fit_temperature(
    probs: Sequence[float], labels: Sequence[int], *, steps: int = 200, t_max: float = 5.0
) -> Calibrator:
    """Grid-search temperature on logit(prob) minimizing NLL. Deterministic; T in [0.05, t_max]."""
    logits = [_logit(p) for p in probs]
    best_t, best = 1.0, math.inf
    for i in range(1, steps + 1):
        t = round(t_max * i / steps, 4)
        loss = 0.0
        for z, y in zip(logits, labels, strict=True):
            q = _clip(_sigmoid(z / t))
            loss -= y * math.log(q) + (1 - y) * math.log(1.0 - q)
        if loss < best - 1e-12:
            best_t, best = t, loss
    return Calibrator(method="temperature", params={"temperature": best_t})


def fit_platt(probs: Sequence[float], labels: Sequence[int], *, iters: int = 100) -> Calibrator:
    """1-D logistic (Platt) on x = logit(prob): q = sigmoid(a*x + b), Newton/IRLS, deterministic."""
    xs = [_logit(p) for p in probs]
    ys = [float(y) for y in labels]
    a, b = 1.0, 0.0
    for _ in range(iters):
        g_a = g_b = h_aa = h_ab = h_bb = 0.0
        for x, y in zip(xs, ys, strict=True):
            q = _sigmoid(a * x + b)
            w = max(q * (1.0 - q), _EPS)
            r = q - y
            g_a += r * x
            g_b += r
            h_aa += w * x * x
            h_ab += w * x
            h_bb += w
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < _EPS:
            break
        da = (g_a * h_bb - g_b * h_ab) / det
        db = (g_b * h_aa - g_a * h_ab) / det
        a -= da
        b -= db
        if abs(da) < 1e-10 and abs(db) < 1e-10:
            break
    return Calibrator(method="platt", params={"a": round(a, 8), "b": round(b, 8)})


def fit_isotonic(
    probs: Sequence[float], labels: Sequence[int], *, min_obs: int = 50
) -> Calibrator | None:
    """Isotonic regression (PAVA) on (prob -> label). Returns None below ``min_obs`` (§15)."""
    n = len(probs)
    if n < min_obs:
        return None
    order = sorted(range(n), key=lambda i: (probs[i], labels[i]))
    xs = [float(probs[i]) for i in order]
    ys = [float(labels[i]) for i in order]
    # Pool-adjacent-violators: weighted means over pooled blocks.
    vals = list(ys)
    weights = [1.0] * n
    i = 0
    while i < len(vals) - 1:
        if vals[i] > vals[i + 1] + 1e-15:
            w = weights[i] + weights[i + 1]
            v = (vals[i] * weights[i] + vals[i + 1] * weights[i + 1]) / w
            vals[i : i + 2] = [v]
            weights[i : i + 2] = [w]
            xs_block = i  # noqa: F841 (kept for clarity; x handled below)
            if i > 0:
                i -= 1
        else:
            i += 1
    # Expand pooled values back to per-observation, then collapse to breakpoints (x = mean prob).
    fitted: list[float] = []
    for v, w in zip(vals, weights, strict=True):
        fitted.extend([v] * round(w))
    bx: list[float] = []
    by: list[float] = []
    for x, y in zip(xs, fitted, strict=True):
        if bx and abs(by[-1] - y) < 1e-15:
            continue
        bx.append(x)
        by.append(y)
    return Calibrator(
        method="isotonic", params={"breakpoints": len(bx)}, _iso_x=tuple(bx), _iso_y=tuple(by)
    )


def demo() -> None:
    """Self-check: calibrators reduce NLL on a miscalibrated, separable-ish synthetic set."""
    probs = [0.2, 0.25, 0.3, 0.35, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85] * 6
    labels = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1] * 6
    raw = nll(probs, labels)
    for cal in (fit_temperature(probs, labels), fit_platt(probs, labels)):
        assert nll(cal.apply(probs), labels) <= raw + 1e-9, cal.method
    iso = fit_isotonic(probs, labels)
    assert iso is not None and nll(iso.apply(probs), labels) <= raw + 1e-9
    # temperature is monotone in the score, so ranking (hence recall/FP curve) is preserved
    t = fit_temperature(probs, labels).apply([0.4, 0.6])
    assert t[0] < t[1]
    print("calibration.fit demo OK")


__all__ = [
    "Calibrator",
    "fit_isotonic",
    "fit_platt",
    "fit_temperature",
    "identity",
    "nll",
]


if __name__ == "__main__":
    demo()
