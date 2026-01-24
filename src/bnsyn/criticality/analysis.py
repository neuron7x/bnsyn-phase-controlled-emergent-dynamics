"""Criticality analysis utilities (branching ratios and power laws)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PowerLawFit:
    """Parameters for a fitted power-law distribution."""

    alpha: float
    xmin: float


def mr_branching_ratio(activity: np.ndarray, max_lag: int = 5) -> float:
    """Estimate the branching ratio using multistep regression.

    Parameters
    ----------
    activity
        Non-negative activity time series ``A(t)``.
    max_lag
        Maximum lag for the regression estimator.

    Returns
    -------
    float
        Estimated branching ratio.

    Raises
    ------
    ValueError
        If inputs are invalid or the estimator cannot be computed.
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
    """Fit a continuous power-law using MLE with fixed ``xmin``.

    Parameters
    ----------
    data
        One-dimensional sample data.
    xmin
        Lower bound for the fit.

    Returns
    -------
    PowerLawFit
        Fitted power-law parameters.

    Raises
    ------
    ValueError
        If inputs are invalid or insufficient for the fit.
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
