from __future__ import annotations

import math

import pytest

from bnsyn import api
from bnsyn.sim.network import run_simulation


def test_run_simulation_fuzz_resolves_to_controlled_error_or_finite_metrics() -> None:
    hypothesis = pytest.importorskip("hypothesis")
    given = hypothesis.given
    st = hypothesis.strategies

    @given(
        steps=st.integers(min_value=-5, max_value=12),
        dt_ms=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        seed=st.integers(min_value=0, max_value=1000),
        n=st.integers(min_value=0, max_value=512),
    )
    def _case(steps: int, dt_ms: float, seed: int, n: int) -> None:
        try:
            metrics = run_simulation(steps=steps, dt_ms=dt_ms, seed=seed, N=n)
        except (TypeError, ValueError, RuntimeError):
            return

        assert set(metrics) == {"sigma_mean", "rate_mean_hz", "sigma_std", "rate_std"}
        assert all(math.isfinite(float(value)) for value in metrics.values())

    _case()


def test_api_run_fuzz_resolves_to_controlled_error_or_finite_metrics() -> None:
    hypothesis = pytest.importorskip("hypothesis")
    given = hypothesis.given
    st = hypothesis.strategies

    @given(
        steps=st.integers(min_value=-5, max_value=10),
        dt_ms=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        seed=st.integers(min_value=0, max_value=1000),
        n=st.integers(min_value=0, max_value=512),
    )
    def _case(steps: int, dt_ms: float, seed: int, n: int) -> None:
        payload = {"steps": steps, "dt_ms": dt_ms, "seed": seed, "N": n}
        try:
            metrics = api.run(payload)
        except (TypeError, ValueError, RuntimeError):
            return

        assert all(math.isfinite(float(value)) for value in metrics.values())

    _case()


def test_run_simulation_rejects_extreme_large_n() -> None:
    with pytest.raises(ValueError, match="MAX_SAFE_NEURONS"):
        run_simulation(steps=1, dt_ms=0.1, seed=1, N=10_000)
