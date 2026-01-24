"""Energy regularization utilities for BN-Syn reward shaping."""

from __future__ import annotations

import numpy as np

from bnsyn.config import EnergyParams


def energy_cost(rate_hz: np.ndarray, w: np.ndarray, I_ext_pA: np.ndarray, p: EnergyParams) -> float:
    """Compute the weighted energy cost for activity and weights.

    Parameters
    ----------
    rate_hz
        Instantaneous firing rates in Hz.
    w
        Synaptic weights.
    I_ext_pA
        External current in pA.
    p
        Energy regularization parameters.

    Returns
    -------
    float
        Scalar energy cost.
    """
    E_rate = float(p.lambda_rate) * float(np.sum(rate_hz**2))
    E_w = float(p.lambda_weight) * float(np.sum(w**2))
    E_stim = float(np.sum(I_ext_pA**2))
    return float(E_rate + E_w + E_stim)


def total_reward(R_task: float, E_total: float, rate_mean_hz: float, p: EnergyParams) -> float:
    """Combine task reward and energy regularization.

    Parameters
    ----------
    R_task
        Task-specific reward.
    E_total
        Total energy cost.
    rate_mean_hz
        Mean firing rate in Hz.
    p
        Energy regularization parameters.

    Returns
    -------
    float
        Reward adjusted by energy penalties and activity floor.
    """
    activity_floor = float(min(rate_mean_hz, float(p.r_min_hz)))
    return float(R_task - float(p.lambda_energy) * float(E_total) + activity_floor)
