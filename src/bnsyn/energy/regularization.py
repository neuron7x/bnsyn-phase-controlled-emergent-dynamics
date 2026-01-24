from __future__ import annotations

import numpy as np

from bnsyn.config import EnergyParams


def energy_cost(rate_hz: np.ndarray, w: np.ndarray, I_ext_pA: np.ndarray, p: EnergyParams) -> float:
    E_rate = float(p.lambda_rate) * float(np.sum(rate_hz**2))
    E_w = float(p.lambda_weight) * float(np.sum(w**2))
    E_stim = float(np.sum(I_ext_pA**2))
    return float(E_rate + E_w + E_stim)


def total_reward(R_task: float, E_total: float, rate_mean_hz: float, p: EnergyParams) -> float:
    activity_floor = float(min(rate_mean_hz, float(p.r_min_hz)))
    return float(R_task - float(p.lambda_energy) * float(E_total) + activity_floor)
