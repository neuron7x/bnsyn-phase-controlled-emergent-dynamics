"""Energy regularization utilities for BN-Syn.

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
SPEC.md §P1-7

Claims
------
CLM-0021
"""

from __future__ import annotations

import numpy as np

from bnsyn.config import EnergyParams


def energy_cost(rate_hz: np.ndarray, w: np.ndarray, I_ext_pA: np.ndarray, p: EnergyParams) -> float:
    """Compute the quadratic energy regularization term.

    Parameters
    ----------
    rate_hz : numpy.ndarray
        Firing rates (Hz).
    w : numpy.ndarray
        Weight matrix.
    I_ext_pA : numpy.ndarray
        External input currents (pA).
    p : EnergyParams
        Energy regularization parameters.

    Returns
    -------
    float
        Energy cost value.

    Determinism
    -----------
    Deterministic under fixed inputs.

    SPEC
    ----
    SPEC.md §P1-7

    Claims
    ------
    CLM-0021
    """
    E_rate = float(p.lambda_rate) * float(np.sum(rate_hz**2))
    E_w = float(p.lambda_weight) * float(np.sum(w**2))
    E_stim = float(np.sum(I_ext_pA**2))
    return float(E_rate + E_w + E_stim)


def total_reward(R_task: float, E_total: float, rate_mean_hz: float, p: EnergyParams) -> float:
    """Compute total reward with energy penalty and activity floor.

    Parameters
    ----------
    R_task : float
        Task reward term.
    E_total : float
        Total energy cost.
    rate_mean_hz : float
        Mean firing rate (Hz).
    p : EnergyParams
        Energy regularization parameters.

    Returns
    -------
    float
        Total reward value.

    Determinism
    -----------
    Deterministic under fixed inputs.

    SPEC
    ----
    SPEC.md §P1-7

    Claims
    ------
    CLM-0021
    """
    activity_floor = float(min(rate_mean_hz, float(p.r_min_hz)))
    return float(R_task - float(p.lambda_energy) * float(E_total) + activity_floor)
