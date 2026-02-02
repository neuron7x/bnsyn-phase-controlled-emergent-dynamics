"""Tests for run_simulation external current handling."""

from __future__ import annotations

import numpy as np
from pytest import MonkeyPatch

from bnsyn.sim import network


def test_run_simulation_with_external_current() -> None:
    metrics = network.run_simulation(
        steps=2,
        dt_ms=0.1,
        seed=1,
        N=4,
        external_current_pA=1.0,
    )
    assert set(metrics.keys()) == {
        "sigma_mean",
        "rate_mean_hz",
        "sigma_std",
        "rate_std",
    }
    assert all(isinstance(value, float) for value in metrics.values())


def test_run_simulation_adaptive_passes_external_current(monkeypatch: MonkeyPatch) -> None:
    captured: list[np.ndarray] = []

    class FakeNetwork:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.calls = 0

        def step(self, *_args: object, **_kwargs: object) -> dict[str, float]:
            raise AssertionError("step should not be used when adaptive=True")

        def step_adaptive(
            self, *, external_current_pA: np.ndarray | None = None, **_kwargs: object
        ) -> dict[str, float]:
            if external_current_pA is None:
                raise AssertionError("external_current_pA should be provided")
            captured.append(external_current_pA.copy())
            self.calls += 1
            return {"sigma": 1.0, "spike_rate_hz": 2.0}

    monkeypatch.setattr(network, "Network", FakeNetwork)

    metrics = network.run_simulation(
        steps=3,
        dt_ms=0.1,
        seed=1,
        N=4,
        external_current_pA=2.5,
        adaptive=True,
    )

    assert metrics["sigma_mean"] == 1.0
    assert metrics["rate_mean_hz"] == 2.0
    assert len(captured) == 3
    for entry in captured:
        assert entry.shape == (4,)
        assert np.allclose(entry, 2.5)
