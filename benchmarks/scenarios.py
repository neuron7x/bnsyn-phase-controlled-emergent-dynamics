"""Benchmark scenario definitions for deterministic BN-Syn performance runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkScenario:
    """Definition of a deterministic benchmark scenario."""

    scenario_id: str
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
            "scenario_id": self.scenario_id,
            "name": self.name,
            "seed": self.seed,
            "dt_ms": self.dt_ms,
            "steps": self.steps,
            "N_neurons": self.N_neurons,
            "p_conn": self.p_conn,
            "frac_inhib": self.frac_inhib,
            "description": self.description,
        }


SMOKE_BENCH = BenchmarkScenario(
    scenario_id="SCN-001",
    name="smoke-bench",
    seed=4242,
    dt_ms=0.1,
    steps=120,
    N_neurons=64,
    p_conn=0.05,
    frac_inhib=0.2,
    description="Tiny deterministic run for CI microbench",
)

CORE_STEP = BenchmarkScenario(
    scenario_id="SCN-002",
    name="core-step",
    seed=4242,
    dt_ms=0.1,
    steps=800,
    N_neurons=512,
    p_conn=0.05,
    frac_inhib=0.2,
    description="Representative core stepping workload",
)

SCALE_SWEEP = [
    BenchmarkScenario(
        scenario_id="SCN-003",
        name=f"scale-sweep-{n}",
        seed=4242,
        dt_ms=0.1,
        steps=600,
        N_neurons=n,
        p_conn=0.05,
        frac_inhib=0.2,
        description=f"Scale sweep for network size N={n}",
    )
    for n in [256, 512, 1024, 2048]
]

LEGACY_STEPS_SWEEP = [
    BenchmarkScenario(
        scenario_id="SCN-004",
        name=f"steps-sweep-{steps}",
        seed=4242,
        dt_ms=0.1,
        steps=steps,
        N_neurons=512,
        p_conn=0.05,
        frac_inhib=0.2,
        description=f"Legacy steps sweep: steps={steps}",
    )
    for steps in [400, 800, 1600]
]

LEGACY_CONN_SWEEP = [
    BenchmarkScenario(
        scenario_id="SCN-005",
        name=f"conn-sweep-{int(p_conn * 100)}pct",
        seed=4242,
        dt_ms=0.1,
        steps=600,
        N_neurons=512,
        p_conn=p_conn,
        frac_inhib=0.2,
        description=f"Legacy connection sweep: p_conn={p_conn}",
    )
    for p_conn in [0.02, 0.05, 0.1]
]

LEGACY_DT_SWEEP = [
    BenchmarkScenario(
        scenario_id="SCN-006",
        name=f"dt-sweep-{int(dt_ms * 1000)}us",
        seed=4242,
        dt_ms=dt_ms,
        steps=600,
        N_neurons=256,
        p_conn=0.05,
        frac_inhib=0.2,
        description=f"Legacy timestep sweep: dt={dt_ms}ms",
    )
    for dt_ms in [0.05, 0.1, 0.2]
]


SCENARIO_SETS = {
    "micro": [SMOKE_BENCH],
    "full": [SMOKE_BENCH, CORE_STEP, *SCALE_SWEEP],
    "ci_smoke": [SMOKE_BENCH],
    "quick": [SMOKE_BENCH, CORE_STEP],
    "n_sweep": SCALE_SWEEP,
    "steps_sweep": LEGACY_STEPS_SWEEP,
    "conn_sweep": LEGACY_CONN_SWEEP,
    "dt_sweep": LEGACY_DT_SWEEP,
}


def get_scenarios(suite: str = "micro") -> list[BenchmarkScenario]:
    """Get scenarios for a given benchmark suite."""
    if suite not in SCENARIO_SETS:
        raise ValueError(
            f"Unknown suite '{suite}'. Available: {', '.join(SCENARIO_SETS.keys())}"
        )
    return SCENARIO_SETS[suite]
