from bnsyn.sim.network import run_simulation


def test_network_runs() -> None:
    m = run_simulation(steps=200, dt_ms=0.1, seed=7, N=100)
    assert "sigma_mean" in m and "rate_mean_hz" in m
