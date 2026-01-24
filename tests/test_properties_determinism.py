from hypothesis import given, settings
import hypothesis.strategies as st

from bnsyn.sim.network import run_simulation


@given(
    n=st.integers(min_value=10, max_value=500),
    dt=st.floats(min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=50, deadline=None)
def test_determinism_property_all_sizes(n: int, dt: float, seed: int) -> None:
    m1 = run_simulation(steps=100, dt_ms=dt, seed=seed, N=n)
    m2 = run_simulation(steps=100, dt_ms=dt, seed=seed, N=n)
    assert m1 == m2, f"Determinism failed for N={n}, dt={dt}, seed={seed}"


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=20, deadline=None)
def test_seed_affects_output(seed: int) -> None:
    m1 = run_simulation(steps=100, dt_ms=0.1, seed=seed, N=100)
    m2 = run_simulation(steps=100, dt_ms=0.1, seed=seed + 1, N=100)
    assert m1 != m2, "Different seeds must produce different results"
