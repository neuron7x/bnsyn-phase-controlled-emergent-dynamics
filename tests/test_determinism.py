from bnsyn.sim.network import run_simulation


def test_determinism_same_seed_same_metrics() -> None:
    m1 = run_simulation(steps=500, dt_ms=0.1, seed=42, N=80)
    m2 = run_simulation(steps=500, dt_ms=0.1, seed=42, N=80)
    assert m1 == m2
