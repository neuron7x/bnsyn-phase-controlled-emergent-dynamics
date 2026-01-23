from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from bnsyn.config import AdExParams
from bnsyn.numerics.integrators import clamp_exp_arg


@dataclass
class AdExState:
    V_mV: np.ndarray  # shape (N,)
    w_pA: np.ndarray  # shape (N,)
    spiked: np.ndarray  # bool, shape (N,)


def adex_step(
    state: AdExState,
    params: AdExParams,
    dt_ms: float,
    I_syn_pA: np.ndarray,
    I_ext_pA: np.ndarray,
) -> AdExState:
    """One explicit Euler step for AdEx with spike-reset and exponential clamp.

    Equations follow Brette & Gerstner (2005) with standard reset:
      - if V > Vpeak: V <- Vreset, w <- w + b
    """
    if dt_ms <= 0:
        raise ValueError("dt_ms must be positive")
    V = state.V_mV.astype(float, copy=True)
    w = state.w_pA.astype(float, copy=True)

    # Convert conductances/currents: parameters are in pF/nS/mV/ms/pA so ms is consistent.
    # dV/dt = ( -gL(V-EL) + gL*DeltaT*exp((V-VT)/DeltaT) - w - I_syn + I_ext ) / C
    exp_arg = (V - params.VT_mV) / params.DeltaT_mV
    exp_arg = np.minimum(exp_arg, 20.0)  # prevent overflow
    I_exp = params.gL_nS * params.DeltaT_mV * np.exp(exp_arg)  # nS*mV ~ pA
    dV = (
        -params.gL_nS * (V - params.EL_mV)
        + I_exp
        - w
        - I_syn_pA
        + I_ext_pA
    ) / params.C_pF
    V = V + dt_ms * dV

    # dw/dt = ( a(V-EL) - w ) / tauw
    dw = (params.a_nS * (V - params.EL_mV) - w) / params.tauw_ms
    w = w + dt_ms * dw

    spiked = V >= params.Vpeak_mV
    if np.any(spiked):
        V[spiked] = params.Vreset_mV
        w[spiked] = w[spiked] + params.b_pA

    return AdExState(V_mV=V, w_pA=w, spiked=spiked)
