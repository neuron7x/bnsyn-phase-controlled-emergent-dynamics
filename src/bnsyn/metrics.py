"""Metrics extraction utilities for BN-Syn simulation state.

Parameters
----------
None

Returns
-------
None

Notes
-----
Provides deterministic numeric metrics for criticality, entropy, energy,
and temperature phase estimation.

References
----------
docs/SPEC.md
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from bnsyn.config import EnergyParams, TemperatureParams
from bnsyn.criticality.analysis import fit_power_law_mle, mr_branching_ratio
from bnsyn.energy.regularization import energy_cost
from bnsyn.temperature.schedule import gate_sigmoid


def _as_array(state: Mapping[str, object], key: str) -> np.ndarray:
    value = state.get(key)
    if value is None:
        raise KeyError(f"Missing state field: {key}")
    return np.asarray(value, dtype=np.float64)


def branching_ratio_sigma(state: Mapping[str, object]) -> float:
    """Estimate branching ratio sigma from activity time series.

    Parameters
    ----------
    state : Mapping[str, object]
        Raw simulation state containing an ``activity`` array.

    Returns
    -------
    float
        Estimated branching ratio sigma.
    """
    if "sigma_series" in state:
        sigma_series = _as_array(state, "sigma_series")
        if sigma_series.size == 0:
            raise ValueError("Sigma series is empty")
        return float(np.mean(sigma_series))
    activity = _as_array(state, "activity")
    return float(mr_branching_ratio(activity))


def _avalanche_sizes(activity: np.ndarray, silence_threshold: float) -> np.ndarray:
    sizes: list[float] = []
    current = 0.0
    for value in activity:
        if value > silence_threshold:
            current += float(value)
        elif current > 0.0:
            sizes.append(current)
            current = 0.0
    if current > 0.0:
        sizes.append(current)
    return np.asarray(sizes, dtype=np.float64)


def avalanche_powerlaw_fit(state: Mapping[str, object]) -> float:
    """Fit a power-law exponent to avalanche sizes derived from activity.

    Parameters
    ----------
    state : Mapping[str, object]
        Raw simulation state containing an ``activity`` array.

    Returns
    -------
    float
        Power-law alpha parameter.
    """
    activity = _as_array(state, "activity")
    silence_threshold = float(np.percentile(activity, 20.0))
    sizes = _avalanche_sizes(activity, silence_threshold)
    if sizes.size < 5:
        raise ValueError("Insufficient avalanche samples for power-law fit")
    xmin = 2.0
    sizes = sizes[sizes >= xmin]
    if sizes.size < 5:
        raise ValueError("Insufficient avalanche samples above xmin")
    fit = fit_power_law_mle(sizes, xmin=xmin)
    return float(fit.alpha)


def entropy_rate(state: Mapping[str, object]) -> float:
    """Compute Shannon entropy of activity distribution.

    Parameters
    ----------
    state : Mapping[str, object]
        Raw simulation state containing an ``activity`` array.

    Returns
    -------
    float
        Shannon entropy in nats.
    """
    activity = _as_array(state, "activity")
    if activity.size == 0:
        raise ValueError("Activity array is empty")
    values, counts = np.unique(activity, return_counts=True)
    if values.size == 0:
        raise ValueError("Activity array has no values")
    probs = counts.astype(np.float64) / float(np.sum(counts))
    return float(-np.sum(probs * np.log(probs)))


def plasticity_energy(state: Mapping[str, object]) -> float:
    """Compute plasticity energy cost from raw state.

    Parameters
    ----------
    state : Mapping[str, object]
        Raw simulation state containing ``rate_hz``, ``weights``, and ``I_ext_pA``.

    Returns
    -------
    float
        Energy cost (dimensionless).
    """
    rate_hz = _as_array(state, "rate_hz")
    weights = _as_array(state, "weights")
    I_ext_pA = _as_array(state, "I_ext_pA")
    return float(energy_cost(rate_hz, weights, I_ext_pA, EnergyParams()))


def temperature_phase(state: Mapping[str, object]) -> float:
    """Compute mean plasticity gate phase from temperature history.

    Parameters
    ----------
    state : Mapping[str, object]
        Raw simulation state containing a ``temperature`` array.

    Returns
    -------
    float
        Mean plasticity gate value in [0, 1].
    """
    temperature = _as_array(state, "temperature")
    params = TemperatureParams()
    gates = np.asarray([gate_sigmoid(float(T), params.Tc, params.gate_tau) for T in temperature])
    return float(np.mean(gates))
