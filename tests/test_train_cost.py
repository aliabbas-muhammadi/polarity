import pytest

from polarity.train.cost import CostError, CostGuard, estimate


def test_estimate_arithmetic():
    est = estimate(steps=1200, sec_per_step=0.6, hourly_rate_aud=1.05)
    assert est.wall_hours == round(1200 * 0.6 / 3600, 4)
    assert est.cost_aud == round(est.wall_hours * 1.05, 4)


def test_guard_refuses_over_cost_cap():
    guard = CostGuard(max_runtime_hours=1.0, max_estimated_cost_aud=1.0, hourly_rate_aud=1.05)
    # ~0.98h (within the runtime cap) but ~A$1.03 (over the A$1.00 cost cap) -> cost trips first
    with pytest.raises(CostError, match="cost"):
        guard.check_estimate(estimate(3528, 1.0, 1.05))


def test_guard_refuses_over_runtime_cap():
    guard = CostGuard(max_runtime_hours=0.25, max_estimated_cost_aud=1000.0, hourly_rate_aud=1.05)
    with pytest.raises(CostError, match="runtime"):
        guard.check_estimate(estimate(3600, 1.0, 1.05))  # 1.0h > 0.25h


def test_guard_allows_within_caps_and_stops_loss_on_elapsed():
    guard = CostGuard(max_runtime_hours=1.0, max_estimated_cost_aud=15.0, hourly_rate_aud=1.05)
    guard.check_estimate(estimate(1592, 0.5, 1.05))  # ~0.22h, ~A$0.23 -> ok
    guard.check_elapsed(0.5)
    with pytest.raises(CostError, match="runtime cap"):
        guard.check_elapsed(1.5)
