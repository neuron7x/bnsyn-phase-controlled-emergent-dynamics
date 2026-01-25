"""Pytest-benchmark integration for deterministic runtime tracking."""

from __future__ import annotations

from bnsyn.sim.network import run_simulation


def _run_demo() -> None:
    _ = run_simulation(steps=2000, dt_ms=0.1, seed=42, N=200)


def test_demo_runtime(benchmark) -> None:
    benchmark(_run_demo)
