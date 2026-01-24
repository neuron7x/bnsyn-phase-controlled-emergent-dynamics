from __future__ import annotations

import numpy as np

from bnsyn.config import EnergyParams


def energy_cost(rate_hz: np.ndarray, w: np.ndarray, I_ext_pA: np.ndarray, p: EnergyParams) -> float:
    E_rate = float(p.lambda_rate) * float(np.sum(rate_hz**2))
    E_w = float(p.lambda_weight) * float(np.sum(w**2))
    E_stim = float(np.sum(I_ext_pA**2))
    return float(E_rate + E_w + E_stim)


def total_reward(r_task: float, e_total: float, rate_mean_hz: float, p: EnergyParams) -> float:
    """Compute total reward with energy cost and activity floor.

    Args:
        r_task: Task-specific reward.
        e_total: Total energy expenditure.
        rate_mean_hz: Mean network firing rate (Hz).
        p: Energy regularization parameters.

    Returns:
        Total reward after energy penalty and activity floor adjustment.
    """
    activity_floor = float(min(rate_mean_hz, float(p.r_min_hz)))
    return float(r_task - float(p.lambda_energy) * float(e_total) + activity_floor)
