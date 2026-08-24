"""Hard cost + runtime caps for the paid run (brief §20/§26). Pure and deterministic.

The runner refuses to *start* if the bottom-up estimate exceeds either cap, and stops-loss mid-run
if elapsed wall time (hence projected spend) crosses a cap. No wall-clock is read here — callers
pass measured values — so this is fully unit-testable offline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class CostError(RuntimeError):
    """Raised when an estimate or elapsed run would exceed an approved cap."""


@dataclass(frozen=True)
class CostEstimate:
    steps: int
    sec_per_step: float
    wall_hours: float
    cost_aud: float


def estimate(steps: int, sec_per_step: float, hourly_rate_aud: float) -> CostEstimate:
    if steps < 0 or sec_per_step < 0 or hourly_rate_aud < 0:
        raise ValueError("estimate inputs must be non-negative")
    wall_hours = steps * sec_per_step / 3600.0
    return CostEstimate(
        steps=steps,
        sec_per_step=sec_per_step,
        wall_hours=round(wall_hours, 4),
        cost_aud=round(wall_hours * hourly_rate_aud, 4),
    )


@dataclass(frozen=True)
class CostGuard:
    max_runtime_hours: float
    max_estimated_cost_aud: float
    hourly_rate_aud: float

    def check_estimate(self, est: CostEstimate) -> None:
        """Refuse to start if the estimate breaches either cap."""
        if est.wall_hours > self.max_runtime_hours + 1e-9:
            raise CostError(
                f"estimated runtime {est.wall_hours}h exceeds cap {self.max_runtime_hours}h"
            )
        if est.cost_aud > self.max_estimated_cost_aud + 1e-9:
            raise CostError(
                f"estimated cost A${est.cost_aud} exceeds cap A${self.max_estimated_cost_aud}"
            )

    def check_elapsed(self, elapsed_hours: float) -> None:
        """Stop-loss mid-run: raise before the run can cross a cap."""
        if elapsed_hours > self.max_runtime_hours + 1e-9:
            raise CostError(
                f"elapsed {round(elapsed_hours, 4)}h exceeds runtime cap {self.max_runtime_hours}h"
            )
        spent = elapsed_hours * self.hourly_rate_aud
        if spent > self.max_estimated_cost_aud + 1e-9:
            raise CostError(
                f"spend A${round(spent, 4)} exceeds cost cap A${self.max_estimated_cost_aud}"
            )


def demo() -> None:
    est = estimate(steps=1200, sec_per_step=0.5, hourly_rate_aud=0.69)
    assert math.isclose(est.wall_hours, 0.1667, abs_tol=1e-3)
    guard = CostGuard(max_runtime_hours=1.0, max_estimated_cost_aud=15.0, hourly_rate_aud=0.69)
    guard.check_estimate(est)  # within caps -> ok
    guard.check_elapsed(0.5)
    for bad in (estimate(100000, 1.0, 0.69),):
        try:
            guard.check_estimate(bad)
        except CostError:
            pass
        else:  # pragma: no cover
            raise AssertionError("expected CostError")
    print("train.cost demo OK")


__all__ = ["CostError", "CostEstimate", "CostGuard", "estimate"]


if __name__ == "__main__":
    demo()
