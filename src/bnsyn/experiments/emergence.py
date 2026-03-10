"""Orchestration helpers for emergence experiments and artifact generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bnsyn.config import AdExParams, CriticalityParams, SynapseParams
from bnsyn.numerics.time import compute_steps_exact
from bnsyn.rng import seed_all
from bnsyn.sim.network import Network, NetworkParams
from bnsyn.validation import NetworkValidationConfig

RUN_ARTIFACT_FORMAT_VERSION = "1.1.0"


def run_emergence_to_disk(
    N: int,
    dt_ms: float,
    duration_ms: float,
    seed: int,
    external_current_pA: float,
    output_dir: Path,
) -> tuple[dict[str, float], str]:
    """Run emergence simulation and write detailed traces to NPZ."""
    steps = compute_steps_exact(duration_ms, dt_ms)

    _ = NetworkValidationConfig(N=N, dt_ms=dt_ms)
    pack = seed_all(seed)
    net = Network(
        NetworkParams(N=N),
        AdExParams(),
        SynapseParams(),
        CriticalityParams(),
        dt_ms=dt_ms,
        rng=pack.np_rng,
    )

    sigmas: list[float] = []
    rates: list[float] = []
    spike_steps: list[int] = []
    spike_neurons: list[int] = []
    injected_current = (
        np.full(N, external_current_pA, dtype=np.float64)
        if abs(external_current_pA) > 1e-9
        else None
    )

    for step in range(steps):
        metrics = net.step(external_current_pA=injected_current)
        sigmas.append(metrics["sigma"])
        rates.append(metrics["spike_rate_hz"])
        spiked_idx = np.flatnonzero(net.state.spiked)
        if spiked_idx.size > 0:
            spike_steps.extend([step] * int(spiked_idx.size))
            spike_neurons.extend(spiked_idx.astype(int).tolist())

    result_metrics = {
        "sigma_mean": float(np.mean(sigmas)),
        "rate_mean_hz": float(np.mean(rates)),
        "sigma_std": float(np.std(sigmas)),
        "rate_std": float(np.std(rates)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / f"run_{seed}_Iext_{int(external_current_pA)}pA.npz"

    np.savez(
        target_path,
        format_version=RUN_ARTIFACT_FORMAT_VERSION,
        spike_steps=np.asarray(spike_steps, dtype=np.int64),
        spike_neurons=np.asarray(spike_neurons, dtype=np.int64),
        sigma_trace=np.asarray(sigmas, dtype=np.float64),
        rate_trace_hz=np.asarray(rates, dtype=np.float64),
        dt_ms=np.float64(dt_ms),
        steps=np.int64(steps),
        N=np.int64(N),
        seed=np.int64(seed),
        external_current_pA=np.float64(external_current_pA),
    )
    return result_metrics, str(target_path)
