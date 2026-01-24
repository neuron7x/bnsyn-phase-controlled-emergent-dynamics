"""Metrics collection for BN-Syn benchmarks."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import psutil

from benchmarks.scenarios.base import BenchmarkScenario
from bnsyn.config import AdExParams, CriticalityParams, SynapseParams, TemperatureParams
from bnsyn.rng import seed_all
from bnsyn.sim.network import Network, NetworkParams
from bnsyn.temperature.schedule import TemperatureSchedule


@dataclass(frozen=True)
class BenchmarkMetrics:
    """Metrics collected from a single benchmark run."""

    stability_nan_rate: float
    stability_divergence_rate: float
    physics_spike_rate_hz_mean: float
    physics_sigma_mean: float
    physics_sigma_std: float
    learning_weight_entropy: float
    learning_convergence_error: float
    thermostat_temperature_mean: float
    thermostat_exploration_mean: float
    thermostat_temperature_exploration_corr: float
    reproducibility_bitwise_delta: float
    performance_wall_time_sec: float
    performance_peak_rss_mb: float
    performance_per_step_ms: float
    performance_neuron_steps_per_sec: float
    params: dict[str, Any]


def _criticality_params(scenario: BenchmarkScenario) -> CriticalityParams:
    if scenario.sigma_target is None:
        return CriticalityParams()
    return CriticalityParams(sigma_target=float(scenario.sigma_target))


def _temperature_params(scenario: BenchmarkScenario) -> TemperatureParams:
    params = TemperatureParams()
    if scenario.temperature_T0 is not None:
        params.T0 = float(scenario.temperature_T0)
    if scenario.temperature_alpha is not None:
        params.alpha = float(scenario.temperature_alpha)
    if scenario.temperature_Tmin is not None:
        params.Tmin = float(scenario.temperature_Tmin)
    if scenario.temperature_Tc is not None:
        params.Tc = float(scenario.temperature_Tc)
    if scenario.temperature_gate_tau is not None:
        params.gate_tau = float(scenario.temperature_gate_tau)
    return params


def _weight_entropy(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    nonzero = weights[weights > 0.0]
    if nonzero.size == 0:
        return 0.0
    total = float(np.sum(nonzero))
    if total <= 0.0:
        return 0.0
    probs = nonzero / total
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy)


def _bitwise_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return 1.0
    if a.size == 0:
        return 0.0
    a_bits = a.view(np.uint64)
    b_bits = b.view(np.uint64)
    return float(np.count_nonzero(a_bits != b_bits) / a_bits.size)


def _run_series(scenario: BenchmarkScenario) -> dict[str, Any]:
    pack = seed_all(scenario.seed)
    rng = pack.np_rng

    nparams = NetworkParams(
        N=scenario.N_neurons,
        p_conn=scenario.p_conn,
        frac_inhib=scenario.frac_inhib,
    )
    net = Network(
        nparams,
        AdExParams(),
        SynapseParams(),
        _criticality_params(scenario),
        dt_ms=scenario.dt_ms,
        rng=rng,
    )

    sigma_series: list[float] = []
    spike_rate_series: list[float] = []
    gain_series: list[float] = []

    nan_count = 0
    total_count = 0
    divergence_steps = 0

    for _ in range(scenario.steps):
        metrics = net.step()
        sigma_series.append(float(metrics["sigma"]))
        spike_rate_series.append(float(metrics["spike_rate_hz"]))
        gain_series.append(float(metrics["gain"]))

        V = net.state.V_mV
        w = net.state.w_pA
        nan_count += int(np.isnan(V).sum() + np.isnan(w).sum())
        total_count += int(V.size + w.size)
        if not (np.all(np.isfinite(V)) and np.all(np.isfinite(w))):
            divergence_steps += 1

    weights_exc = net.W_exc.to_dense()
    weights_inh = net.W_inh.to_dense()
    weight_entropy = _weight_entropy(np.concatenate([weights_exc.ravel(), weights_inh.ravel()]))

    return {
        "sigma": np.asarray(sigma_series, dtype=np.float64),
        "spike_rate_hz": np.asarray(spike_rate_series, dtype=np.float64),
        "gain": np.asarray(gain_series, dtype=np.float64),
        "nan_count": nan_count,
        "total_count": total_count,
        "divergence_steps": divergence_steps,
        "weight_entropy": weight_entropy,
        "sigma_target": _criticality_params(scenario).sigma_target,
    }


def run_benchmark(scenario: BenchmarkScenario) -> BenchmarkMetrics:
    """Run a single benchmark scenario and compute metrics."""
    process = psutil.Process(os.getpid())
    start_rss = process.memory_info().rss / (1024 * 1024)
    start_time = time.perf_counter()

    series = _run_series(scenario)

    wall_time = time.perf_counter() - start_time
    end_rss = process.memory_info().rss / (1024 * 1024)
    peak_rss = max(start_rss, end_rss)

    sigmas = series["sigma"]
    spike_rates = series["spike_rate_hz"]

    nan_rate = float(series["nan_count"] / series["total_count"]) if series["total_count"] else 0.0
    divergence_rate = float(series["divergence_steps"] / scenario.steps) if scenario.steps else 0.0

    sigma_mean = float(np.mean(sigmas))
    sigma_std = float(np.std(sigmas))
    spike_rate_mean = float(np.mean(spike_rates))

    window = max(1, int(0.1 * scenario.steps))
    sigma_target = float(series["sigma_target"])
    convergence_error = float(abs(np.mean(sigmas[-window:]) - sigma_target))

    temp_params = _temperature_params(scenario)
    schedule = TemperatureSchedule(params=temp_params)
    temperatures = np.zeros(scenario.steps, dtype=np.float64)
    for idx in range(scenario.steps):
        temperatures[idx] = schedule.step_geometric()

    exploration = np.abs(np.diff(spike_rates, prepend=spike_rates[0]))
    temp_std = float(np.std(temperatures))
    exploration_std = float(np.std(exploration))
    if temp_std > 0.0 and exploration_std > 0.0:
        corr = float(np.corrcoef(temperatures, exploration)[0, 1])
    else:
        corr = 0.0

    series_repeat = _run_series(scenario)
    reproducibility_delta = max(
        _bitwise_delta(series["sigma"], series_repeat["sigma"]),
        _bitwise_delta(series["spike_rate_hz"], series_repeat["spike_rate_hz"]),
    )

    neuron_steps = scenario.N_neurons * scenario.steps
    per_step_ms = (wall_time / scenario.steps) * 1000.0
    neuron_steps_per_sec = neuron_steps / wall_time if wall_time > 0.0 else 0.0

    return BenchmarkMetrics(
        stability_nan_rate=nan_rate,
        stability_divergence_rate=divergence_rate,
        physics_spike_rate_hz_mean=spike_rate_mean,
        physics_sigma_mean=sigma_mean,
        physics_sigma_std=sigma_std,
        learning_weight_entropy=float(series["weight_entropy"]),
        learning_convergence_error=convergence_error,
        thermostat_temperature_mean=float(np.mean(temperatures)),
        thermostat_exploration_mean=float(np.mean(exploration)),
        thermostat_temperature_exploration_corr=corr,
        reproducibility_bitwise_delta=reproducibility_delta,
        performance_wall_time_sec=wall_time,
        performance_peak_rss_mb=peak_rss,
        performance_per_step_ms=per_step_ms,
        performance_neuron_steps_per_sec=float(neuron_steps_per_sec),
        params=scenario.to_dict(),
    )


def metrics_to_dict(metrics: BenchmarkMetrics) -> dict[str, Any]:
    """Serialize metrics to dict."""
    return {
        "stability_nan_rate": metrics.stability_nan_rate,
        "stability_divergence_rate": metrics.stability_divergence_rate,
        "physics_spike_rate_hz_mean": metrics.physics_spike_rate_hz_mean,
        "physics_sigma_mean": metrics.physics_sigma_mean,
        "physics_sigma_std": metrics.physics_sigma_std,
        "learning_weight_entropy": metrics.learning_weight_entropy,
        "learning_convergence_error": metrics.learning_convergence_error,
        "thermostat_temperature_mean": metrics.thermostat_temperature_mean,
        "thermostat_exploration_mean": metrics.thermostat_exploration_mean,
        "thermostat_temperature_exploration_corr": (
            metrics.thermostat_temperature_exploration_corr
        ),
        "reproducibility_bitwise_delta": metrics.reproducibility_bitwise_delta,
        "performance_wall_time_sec": metrics.performance_wall_time_sec,
        "performance_peak_rss_mb": metrics.performance_peak_rss_mb,
        "performance_per_step_ms": metrics.performance_per_step_ms,
        "performance_neuron_steps_per_sec": metrics.performance_neuron_steps_per_sec,
    }
