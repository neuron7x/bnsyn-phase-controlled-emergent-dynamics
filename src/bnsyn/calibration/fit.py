"""Calibration utilities for neuron transfer functions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LineFit:
    """Linear fit parameters and fit quality."""

    slope: float
    intercept: float
    r2: float


def fit_line(x: np.ndarray, y: np.ndarray) -> LineFit:
    """Fit a line to data using least squares.

    Parameters
    ----------
    x
        One-dimensional input values.
    y
        One-dimensional target values.

    Returns
    -------
    LineFit
        Fitted line parameters and R2.

    Raises
    ------
    ValueError
        If inputs are not 1D arrays with matching shapes.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("x,y must be 1D arrays of same shape")
    X = np.vstack([x, np.ones_like(x)]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return LineFit(slope=float(beta[0]), intercept=float(beta[1]), r2=float(r2))


def fit_fI_curve(I_pA: np.ndarray, rate_hz: np.ndarray) -> LineFit:
    """Fit a linear approximation to an f-I curve.

    Parameters
    ----------
    I_pA
        Input current values in pA.
    rate_hz
        Firing rate values in Hz.

    Returns
    -------
    LineFit
        Linear fit parameters for the f-I curve.
    """
    return fit_line(I_pA, rate_hz)
