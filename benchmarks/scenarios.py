"""Benchmark scenario definitions for BN-Syn scalability testing.

Defines parameter sweeps for:
- N_neurons (network size)
- Connection density
- Simulation steps
- Integration timestep
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkScenario:
    """Definition of a single benchmark scenario."""

    name: str
    seed: int
    dt_ms: float
    steps: int
    N_neurons: int
    p_conn: float
    frac_inhib: float
    description: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "seed": self.seed,
            "dt_ms": self.dt_ms,
            "steps": self.steps,
            "N_neurons": self.N_neurons,
            "p_conn": self.p_conn,
            "frac_inhib": self.frac_inhib,
            "description": self.description,
        }


# CI smoke test: minimal scenario for fast CI validation
CI_SMOKE = BenchmarkScenario(
    name="ci_smoke",
    seed=42,
    dt_ms=0.1,
    steps=100,
    N_neurons=50,
    p_conn=0.05,
    frac_inhib=0.2,
    description="Minimal scenario for CI smoke test",
)

# N_neurons sweep: scalability with network size (fixed steps)
N_SWEEP_SMALL = [
    BenchmarkScenario(
        name=f"n_sweep_{n}",
        seed=42,
        dt_ms=0.1,
        steps=500,
        N_neurons=n,
        p_conn=0.05,
        frac_inhib=0.2,
        description=f"Network size sweep: N={n}",
    )
    for n in [100, 200, 500, 1000]
]

N_SWEEP_LARGE = [
    BenchmarkScenario(
        name=f"n_sweep_{n}",
        seed=42,
        dt_ms=0.1,
        steps=500,
        N_neurons=n,
        p_conn=0.05,
        frac_inhib=0.2,
        description=f"Network size sweep: N={n}",
    )
    for n in [2000, 5000, 10000]
]

# Steps sweep: scalability with simulation length (fixed N)
STEPS_SWEEP = [
    BenchmarkScenario(
        name=f"steps_sweep_{s}",
        seed=42,
        dt_ms=0.1,
        steps=s,
        N_neurons=500,
        p_conn=0.05,
        frac_inhib=0.2,
        description=f"Steps sweep: steps={s}",
    )
    for s in [500, 1000, 2000, 5000]
]

# Connectivity density sweep
CONN_SWEEP = [
    BenchmarkScenario(
        name=f"conn_sweep_{int(p*100)}pct",
        seed=42,
        dt_ms=0.1,
        steps=500,
        N_neurons=500,
        p_conn=p,
        frac_inhib=0.2,
        description=f"Connectivity sweep: p_conn={p}",
    )
    for p in [0.01, 0.05, 0.1, 0.2]
]

# Timestep sweep (dt invariance check)
DT_SWEEP = [
    BenchmarkScenario(
        name=f"dt_sweep_{int(dt*1000)}us",
        seed=42,
        dt_ms=dt,
        steps=1000,
        N_neurons=200,
        p_conn=0.05,
        frac_inhib=0.2,
        description=f"Timestep sweep: dt={dt}ms",
    )
    for dt in [0.05, 0.1, 0.2, 0.5]
]


# Scenario sets for different benchmark runs
SCENARIO_SETS = {
    "ci_smoke": [CI_SMOKE],
    "quick": [CI_SMOKE] + N_SWEEP_SMALL[:2] + STEPS_SWEEP[:2],
    "n_sweep": N_SWEEP_SMALL + N_SWEEP_LARGE,
    "steps_sweep": STEPS_SWEEP,
    "conn_sweep": CONN_SWEEP,
    "dt_sweep": DT_SWEEP,
    "full": N_SWEEP_SMALL + N_SWEEP_LARGE + STEPS_SWEEP + CONN_SWEEP + DT_SWEEP,
}


def get_scenarios(scenario_set: str = "quick") -> list[BenchmarkScenario]:
    """Get scenarios for a given benchmark set."""
    if scenario_set not in SCENARIO_SETS:
        raise ValueError(
            f"Unknown scenario set '{scenario_set}'. "
            f"Available: {', '.join(SCENARIO_SETS.keys())}"
        )
    return SCENARIO_SETS[scenario_set]
