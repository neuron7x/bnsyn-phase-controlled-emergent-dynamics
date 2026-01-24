"""Calibration utilities for f-I curve fitting.

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
SPEC.md §P2-10

Claims
------
CLM-0024
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LineFit:
    """Linear fit parameters and coefficient of determination.

    Parameters
    ----------
    slope : float
        Linear slope.
    intercept : float
        Linear intercept.
    r2 : float
        Coefficient of determination.

    Returns
    -------
    LineFit
        Fit parameter container.

    Determinism
    -----------
    Deterministic under fixed inputs.

    SPEC
    ----
    SPEC.md §P2-10

    Claims
    ------
    CLM-0024
    """
    slope: float
    intercept: float
    r2: float


def fit_line(x: np.ndarray, y: np.ndarray) -> LineFit:
    """Fit a line to data using least squares.

    Parameters
    ----------
    x : numpy.ndarray
        Input values.
    y : numpy.ndarray
        Output values.

    Returns
    -------
    LineFit
        Linear fit parameters.

    Determinism
    -----------
    Deterministic under fixed inputs.

    SPEC
    ----
    SPEC.md §P2-10

    Claims
    ------
    CLM-0024
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
    """Fit a linear f-I curve in the linear response regime.

    Parameters
    ----------
    I_pA : numpy.ndarray
        Input current values (pA).
    rate_hz : numpy.ndarray
        Output firing rates (Hz).

    Returns
    -------
    LineFit
        Linear fit parameters for the f-I curve.

    Determinism
    -----------
    Deterministic under fixed inputs.

    SPEC
    ----
    SPEC.md §P2-10

    Claims
    ------
    CLM-0024
    """
    return fit_line(I_pA, rate_hz)
