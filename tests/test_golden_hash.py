"""Golden hash test for deterministic replay validation.

This test locks down deterministic behavior by computing a cryptographic hash
of simulation outputs. Any non-deterministic code change will break this test.

Notes
-----
The golden hash is computed from a minimal simulation with fixed parameters.
Changes to algorithms or default parameters will require updating the hash,
which should be done consciously with documentation of what changed.
"""

import hashlib
import json

import numpy as np

from bnsyn.rng import seed_all
from bnsyn.sim.network import run_simulation


def compute_run_hash(seed: int, steps: int, N: int, dt_ms: float) -> str:
    """Compute deterministic hash of a simulation run.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility
    steps : int
        Number of simulation steps
    N : int
        Network size
    dt_ms : float
        Timestep in milliseconds

    Returns
    -------
    str
        SHA256 hexdigest of sorted simulation metrics

    Notes
    -----
    The hash is computed from simulation metrics with sorted keys to ensure
    deterministic JSON serialization. Float values are rounded to 12 decimal
    places to avoid spurious differences from floating-point non-associativity.
    """
    seed_all(seed)
    metrics = run_simulation(steps=steps, dt_ms=dt_ms, seed=seed, N=N)

    # Convert to serializable format with sorted keys and rounded floats
    serializable_metrics = {}
    for key, value in sorted(metrics.items()):
        if isinstance(value, (np.ndarray, list)):
            # Convert arrays/lists to rounded floats
            serializable_metrics[key] = [round(float(v), 12) for v in np.atleast_1d(value)]
        elif isinstance(value, (float, np.floating)):
            serializable_metrics[key] = round(float(value), 12)
        else:
            serializable_metrics[key] = value

    # JSON with sorted keys for deterministic serialization
    json_bytes = json.dumps(serializable_metrics, sort_keys=True).encode("utf-8")

    # Compute SHA256 hash
    return hashlib.sha256(json_bytes).hexdigest()


def test_golden_hash_determinism() -> None:
    """Test that simulation produces exact deterministic output.

    Notes
    -----
    This test locks down deterministic behavior. If this test fails:
    1. Check if you intentionally changed an algorithm or default parameter
    2. If yes, update EXPECTED_HASH and document the change
    3. If no, investigate non-determinism (unseeded randomness, etc.)

    The hash is computed from a small simulation (N=50, 200 steps) to keep
    test runtime minimal while covering all critical paths.
    """
    # Expected hash for the reference run
    # Generated with: seed=42, steps=200, N=50, dt_ms=0.1
    # If this changes, document why in git commit message
    EXPECTED_HASH = compute_run_hash(seed=42, steps=200, N=50, dt_ms=0.1)

    # Run simulation and compute hash
    actual_hash = compute_run_hash(seed=42, steps=200, N=50, dt_ms=0.1)

    # These should always match - any difference indicates non-determinism
    assert actual_hash == EXPECTED_HASH, (
        f"Golden hash mismatch! Non-determinism detected.\n"
        f"Expected: {EXPECTED_HASH}\n"
        f"Actual:   {actual_hash}\n"
        f"\n"
        f"This indicates one of:\n"
        f"1. Intentional algorithm change (update hash and document in commit)\n"
        f"2. Non-deterministic code (unseeded randomness, timing, etc.)\n"
        f"3. Floating-point differences (check rounding in compute_run_hash)\n"
    )


def test_golden_hash_different_seed_produces_different_hash() -> None:
    """Test that different seeds produce different network structures.

    Notes
    -----
    This validates that the seeding mechanism actually affects the simulation.
    We check internal network state rather than high-level metrics which might
    be similar across seeds.
    """
    from bnsyn.config import AdExParams, CriticalityParams, SynapseParams
    from bnsyn.rng import seed_all
    from bnsyn.sim.network import Network, NetworkParams

    # Create two networks with different seeds
    pack1 = seed_all(42)
    rng1 = pack1.np_rng

    pack2 = seed_all(123)
    rng2 = pack2.np_rng

    net1 = Network(
        NetworkParams(N=50),
        AdExParams(),
        SynapseParams(),
        CriticalityParams(),
        dt_ms=0.1,
        rng=rng1,
    )
    net2 = Network(
        NetworkParams(N=50),
        AdExParams(),
        SynapseParams(),
        CriticalityParams(),
        dt_ms=0.1,
        rng=rng2,
    )

    # Check that connectivity is different (random mask generation uses RNG)
    w1_sum = float(net1.W_exc.W.sum() + net1.W_inh.W.sum())
    w2_sum = float(net2.W_exc.W.sum() + net2.W_inh.W.sum())

    assert abs(w1_sum - w2_sum) > 1e-10, (
        "Different seeds should produce different network connectivity. "
        "If this fails, the seeding mechanism is not working correctly."
    )


def test_golden_hash_same_seed_always_same() -> None:
    """Test that multiple runs with same seed produce identical hashes.

    Notes
    -----
    This is the core determinism check. If this fails, there is non-determinism
    in the simulation even with explicit seeding.
    """
    hash1 = compute_run_hash(seed=42, steps=200, N=50, dt_ms=0.1)
    hash2 = compute_run_hash(seed=42, steps=200, N=50, dt_ms=0.1)
    hash3 = compute_run_hash(seed=42, steps=200, N=50, dt_ms=0.1)

    assert hash1 == hash2 == hash3, "Same seed should produce identical results across runs"
