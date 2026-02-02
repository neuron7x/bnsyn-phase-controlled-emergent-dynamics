"""Deterministic regression test for run_simulation metrics."""

from __future__ import annotations

import pytest

from bnsyn.sim.network import run_simulation


def test_run_simulation_regression_metrics() -> None:
    metrics = run_simulation(steps=200, dt_ms=0.1, seed=123, N=60, external_current_pA=300.0)

    assert metrics["sigma_mean"] == pytest.approx(1.0337865415484362, rel=1e-9)
    assert metrics["sigma_std"] == pytest.approx(0.05122639600288859, rel=1e-9)
    assert metrics["rate_mean_hz"] == pytest.approx(50.83333333333333, rel=1e-9)
    assert metrics["rate_std"] == pytest.approx(109.5413417644478, rel=1e-9)
