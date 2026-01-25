"""Validation tests for physical invariants and phase metrics."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import psutil
import pytest

from bnsyn.config import AdExParams, CriticalityParams, SynapseParams, TemperatureParams
from bnsyn.metrics import (
    avalanche_powerlaw_fit,
    branching_ratio_sigma,
    entropy_rate,
    plasticity_energy,
    temperature_phase,
)
from bnsyn.rng import seed_all
from bnsyn.sim.network import Network, NetworkParams
from bnsyn.temperature.schedule import TemperatureSchedule


def _simulate_state(
    *,
    steps: int,
    dt_ms: float,
    seed: int,
    n_neurons: int,
    temperature_params: TemperatureParams,
    ext_rate_hz: float,
    ext_w_nS: float,
) -> dict[str, np.ndarray | float]:
    pack = seed_all(seed)
    rng = pack.np_rng
    net = Network(
        NetworkParams(N=n_neurons, ext_rate_hz=ext_rate_hz, ext_w_nS=ext_w_nS),
        AdExParams(),
        SynapseParams(),
        CriticalityParams(),
        dt_ms=dt_ms,
        rng=rng,
    )
    activity = np.zeros(steps, dtype=np.float64)
    rates = np.zeros(steps, dtype=np.float64)
    gains = np.zeros(steps, dtype=np.float64)
    sigmas = np.zeros(steps, dtype=np.float64)
    for idx in range(steps):
        metrics = net.step()
        activity[idx] = metrics["A_t1"]
        rates[idx] = metrics["spike_rate_hz"]
        gains[idx] = metrics["gain"]
        sigmas[idx] = metrics["sigma"]

    weights = np.zeros((n_neurons, n_neurons), dtype=np.float64)
    weights[:, : net.nE] = net.W_exc.to_dense()
    weights[:, net.nE :] = net.W_inh.to_dense()
    mean_rate = float(np.mean(rates))
    rate_hz = np.full(n_neurons, mean_rate, dtype=np.float64)
    mean_gain = float(np.mean(gains))
    I_ext_pA = np.full(n_neurons, 50.0 * (mean_gain - 1.0), dtype=np.float64)

    schedule = TemperatureSchedule(temperature_params)
    temperatures = np.zeros(steps, dtype=np.float64)
    for idx in range(steps):
        temperatures[idx] = schedule.step_geometric()

    return {
        "activity": activity,
        "rate_hz": rate_hz,
        "weights": weights,
        "I_ext_pA": I_ext_pA,
        "temperature": temperatures,
        "sigma_series": sigmas,
    }


def _temperature_activity(
    *,
    steps: int,
    seed: int,
    temperature_params: TemperatureParams,
) -> dict[str, np.ndarray | float]:
    rng = np.random.default_rng(seed)
    schedule = TemperatureSchedule(temperature_params)
    temperatures = np.zeros(steps, dtype=np.float64)
    activity = np.zeros(steps, dtype=np.float64)
    for idx in range(steps):
        temperature = schedule.step_geometric()
        temperatures[idx] = temperature
        activity[idx] = float(rng.poisson(lam=float(temperature)))
    return {"activity": activity, "temperature": temperatures}


def _write_metrics(metrics: dict[str, float]) -> None:
    output_path = Path("artifacts/physics_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "{\n"
        + ",\n".join(f"  \"{key}\": {value}" for key, value in metrics.items())
        + "\n}\n"
    )


@pytest.mark.validation
def test_physical_invariants() -> None:
    steps = 2000
    dt_ms = 0.1
    seed = 21
    n_neurons = 200
    temp_params = TemperatureParams()

    start = time.perf_counter()
    state = _simulate_state(
        steps=steps,
        dt_ms=dt_ms,
        seed=seed,
        n_neurons=n_neurons,
        temperature_params=temp_params,
        ext_rate_hz=1000.0,
        ext_w_nS=1.0,
    )
    runtime_ms = (time.perf_counter() - start) * 1000.0
    process = psutil.Process()
    memory_mb = process.memory_info().rss / (1024.0 * 1024.0)

    sigma = branching_ratio_sigma(state)
    alpha_state = _simulate_state(
        steps=steps,
        dt_ms=dt_ms,
        seed=seed,
        n_neurons=n_neurons,
        temperature_params=temp_params,
        ext_rate_hz=500.0,
        ext_w_nS=5.0,
    )
    alpha = avalanche_powerlaw_fit(alpha_state)
    entropy = entropy_rate(state)
    energy = plasticity_energy(state)
    phase = temperature_phase(state)

    assert 0.95 <= sigma <= 1.05
    assert 1.5 <= alpha <= 2.5
    assert energy >= 0.0
    assert energy <= 1.0e6

    low_temp = TemperatureParams(T0=0.5, Tmin=1e-3, alpha=0.95, Tc=0.1, gate_tau=0.02)
    high_temp = TemperatureParams(T0=2.0, Tmin=1e-3, alpha=0.95, Tc=0.1, gate_tau=0.02)
    entropy_low = entropy_rate(
        _temperature_activity(steps=steps, seed=seed, temperature_params=low_temp)
    )
    entropy_high = entropy_rate(
        _temperature_activity(steps=steps, seed=seed, temperature_params=high_temp)
    )
    assert entropy_high > entropy_low

    from bnsyn.sim.network import run_simulation

    m1 = run_simulation(steps=steps, dt_ms=dt_ms, seed=seed, N=n_neurons)
    m2 = run_simulation(steps=steps * 2, dt_ms=dt_ms / 2.0, seed=seed, N=n_neurons)
    dt_error = abs(m1["sigma_mean"] - m2["sigma_mean"]) / max(m1["sigma_mean"], 1e-6)
    assert dt_error < 0.01

    _write_metrics(
        {
            "sigma": sigma,
            "entropy": entropy,
            "powerlaw_alpha": alpha,
            "plasticity_energy": energy,
            "temperature": phase,
            "dt_error": dt_error,
            "runtime_ms": runtime_ms,
            "memory_mb": memory_mb,
        }
    )
