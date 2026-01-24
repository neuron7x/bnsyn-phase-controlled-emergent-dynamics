"""Offline criticality analysis utilities.

Parameters
----------
None

Returns
-------
None

Determinism
-----------
Deterministic under fixed inputs.

SPEC
----
SPEC.md §P0-4

Claims
------
CLM-0008, CLM-0009
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PowerLawFit:
    """Power-law fit parameters for alpha and xmin.

    Parameters
    ----------
    alpha : float
        Power-law exponent.
    xmin : float
        Minimum value used for the fit.

    Returns
    -------
    PowerLawFit
        Fit parameter container.

    Determinism
    -----------
    Deterministic given fixed inputs.

    SPEC
    ----
    SPEC.md §P0-4

    Claims
    ------
    CLM-0009
    """
    alpha: float
    xmin: float


def mr_branching_ratio(activity: np.ndarray, max_lag: int = 5) -> float:
    """Estimate branching ratio using a multistep regression approach.

    Parameters
    ----------
    activity : numpy.ndarray
        1D activity counts per time bin.
    max_lag : int
        Maximum regression lag.

    Returns
    -------
    float
        Estimated branching ratio.

    Determinism
    -----------
    Deterministic under fixed inputs.

    SPEC
    ----
    SPEC.md §P0-4

    Claims
    ------
    CLM-0008
    """
    if activity.ndim != 1:
        raise ValueError("activity must be 1D")
    if len(activity) <= max_lag + 1:
        raise ValueError("activity length too short for max_lag")
    if np.any(activity < 0):
        raise ValueError("activity must be non-negative")

    sigma_estimates: list[float] = []
    for k in range(1, max_lag + 1):
        x = activity[:-k]
        y = activity[k:]
        denom = float(np.dot(x, x))
        if denom == 0.0:
            continue
        slope = float(np.dot(x, y) / denom)
        if slope <= 0.0:
            continue
        sigma_estimates.append(slope ** (1.0 / k))
    if not sigma_estimates:
        raise ValueError("unable to estimate branching ratio")
    return float(np.mean(sigma_estimates))


def fit_power_law_mle(data: np.ndarray, xmin: float) -> PowerLawFit:
    """Continuous power-law MLE fit for alpha with fixed xmin.

    Parameters
    ----------
    data : numpy.ndarray
        1D sample data for the fit.
    xmin : float
        Minimum value to include in the fit.

    Returns
    -------
    PowerLawFit
        Fit parameters for the power-law model.

    Determinism
    -----------
    Deterministic under fixed inputs.

    SPEC
    ----
    SPEC.md §P0-4

    Claims
    ------
    CLM-0009
    """
    if data.ndim != 1:
        raise ValueError("data must be 1D")
    if xmin <= 0:
        raise ValueError("xmin must be positive")
    if np.any(data < xmin):
        raise ValueError("data contains values below xmin")
    logs = np.log(data / xmin)
    if np.all(logs == 0):
        raise ValueError("data must include values above xmin for power-law fit")
    alpha = 1.0 + len(data) / float(np.sum(logs))
    return PowerLawFit(alpha=float(alpha), xmin=float(xmin))
