"""Performance regression tests with timing assertions.

These tests ensure that critical operations remain performant
and don't degrade over time.
"""

import time

import numpy as np
import pytest

from bnsyn.config import AdExParams, SynapseParams, CriticalityParams
from bnsyn.neuron.adex import AdExState, adex_step
from bnsyn.rng import seed_all
from bnsyn.sim.network import Network, NetworkParams
from bnsyn.synapse.conductance import nmda_mg_block


@pytest.mark.performance
def test_adex_step_performance() -> None:
    """Test AdEx step performance for N=1000 neurons."""
    # Setup
    rng_pack = seed_all(42)
    rng = rng_pack.np_rng

    N = 1000
    params = AdExParams()
    state = AdExState(V_mV=np.full(N, -65.0), w_pA=np.zeros(N), spiked=np.zeros(N, dtype=bool))
    I_syn = np.zeros(N)
    I_ext = rng.normal(0, 10.0, size=N)

    # Warmup
    for _ in range(10):
        state = adex_step(state, params, dt_ms=0.1, I_syn_pA=I_syn, I_ext_pA=I_ext)

    # Timed runs
    start = time.perf_counter()
    for _ in range(100):
        state = adex_step(state, params, dt_ms=0.1, I_syn_pA=I_syn, I_ext_pA=I_ext)
    elapsed = time.perf_counter() - start

    # Performance assertion: 100 steps for N=1000 should take < 100ms
    assert elapsed < 0.1, f"AdEx step too slow: {elapsed:.3f}s for 100 steps (N=1000)"

    # Log timing for monitoring
    per_step_us = (elapsed / 100) * 1e6
    print(f"\nAdEx performance: {per_step_us:.1f} µs/step (N={N})")


@pytest.mark.performance
def test_nmda_block_performance() -> None:
    """Test NMDA block computation performance."""
    rng_pack = seed_all(42)
    rng = rng_pack.np_rng

    N = 10000
    V = rng.normal(-60.0, 20.0, size=N)  # Voltages around -60mV

    # Warmup
    for _ in range(10):
        _ = nmda_mg_block(V, mg_mM=1.0)

    # Timed runs
    start = time.perf_counter()
    for _ in range(1000):
        _ = nmda_mg_block(V, mg_mM=1.0)
    elapsed = time.perf_counter() - start

    # Performance assertion: 1000 calls for N=10000 should take < 200ms
    assert elapsed < 0.2, f"NMDA block too slow: {elapsed:.3f}s for 1000 calls (N={N})"

    # Log timing
    per_call_us = (elapsed / 1000) * 1e6
    print(f"\nNMDA block performance: {per_call_us:.1f} µs/call (N={N})")


@pytest.mark.performance
@pytest.mark.integration
def test_network_step_performance() -> None:
    """Test Network.step performance for N=200 network."""
    # Setup small network using default params
    rng_pack = seed_all(42)

    nparams = NetworkParams(N=200, frac_inhib=0.2, p_conn=0.05)
    adex = AdExParams()
    syn = SynapseParams()
    crit = CriticalityParams()

    net = Network(
        nparams=nparams,
        adex=adex,
        syn=syn,
        crit=crit,
        dt_ms=0.1,
        rng=rng_pack.np_rng,
        backend="reference",
    )

    # Warmup
    for _ in range(10):
        net.step()

    # Timed runs
    start = time.perf_counter()
    for _ in range(100):
        net.step()
    elapsed = time.perf_counter() - start

    # Performance assertion: 100 steps should take < 1s
    N = nparams.N
    assert elapsed < 1.0, f"Network step too slow: {elapsed:.3f}s for 100 steps (N={N})"

    # Log timing
    per_step_ms = (elapsed / 100) * 1e3
    print(f"\nNetwork performance: {per_step_ms:.2f} ms/step (N={N})")


@pytest.mark.performance
def test_sparse_matmul_performance() -> None:
    """Test sparse matrix multiplication performance."""
    from bnsyn.connectivity import SparseConnectivity

    N = 1000
    p_conn = 0.1
    rng_pack = seed_all(42)
    rng = rng_pack.np_rng

    # Create sparse connectivity - optimized: generate mask first
    mask = rng.random((N, N)) < p_conn
    W_dense = mask.astype(np.float64) * rng.random((N, N))
    W = SparseConnectivity(W_dense, force_format="sparse")

    x = rng.random(N)

    # Warmup
    for _ in range(10):
        _ = W.apply(x)

    # Timed runs
    start = time.perf_counter()
    for _ in range(1000):
        _ = W.apply(x)
    elapsed = time.perf_counter() - start

    # Performance assertion: 1000 sparse matvecs should take < 200ms
    assert elapsed < 0.2, f"Sparse matmul too slow: {elapsed:.3f}s for 1000 ops (N={N})"

    # Log timing
    per_op_us = (elapsed / 1000) * 1e6
    print(f"\nSparse matmul performance: {per_op_us:.1f} µs/op (N={N}, p={p_conn})")


if __name__ == "__main__":
    # Run with: pytest tests/test_performance.py -v -s
    pytest.main([__file__, "-v", "-s", "-m", "performance"])
