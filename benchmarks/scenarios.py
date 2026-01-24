"""Deterministic benchmark scenarios for performance measurements."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioSpec:
    """Immutable scenario definition for deterministic benchmarking."""

    scenario_id: str
    name: str
    seed: int
    steps: int
    dt_ms: float
    n_neurons: int
    p_conn: float
    frac_inhib: float
    sweep_sizes: tuple[int, ...] | None = None


SCENARIO_SMOKE = ScenarioSpec(
    scenario_id="SCN-001",
    name="smoke-bench",
    seed=202401,
    steps=50,
    dt_ms=0.5,
    n_neurons=64,
    p_conn=0.1,
    frac_inhib=0.2,
)

SCENARIO_CORE = ScenarioSpec(
    scenario_id="SCN-002",
    name="core-step",
    seed=202402,
    steps=200,
    dt_ms=0.5,
    n_neurons=256,
    p_conn=0.1,
    frac_inhib=0.2,
)

SCENARIO_SCALE_SWEEP = ScenarioSpec(
    scenario_id="SCN-003",
    name="scale-sweep",
    seed=202403,
    steps=150,
    dt_ms=0.5,
    n_neurons=128,
    p_conn=0.1,
    frac_inhib=0.2,
    sweep_sizes=(128, 256, 512),
)


def get_suite(suite: str) -> list[ScenarioSpec]:
    """Return scenario specs for a benchmark suite."""
    suite_key = suite.lower().strip()
    if suite_key == "micro":
        return [SCENARIO_SMOKE]
    if suite_key == "full":
        return [SCENARIO_SMOKE, SCENARIO_CORE, SCENARIO_SCALE_SWEEP]
    raise ValueError(f"Unknown suite: {suite}")
