from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from bnsyn.config import AdExParams
from bnsyn.validation import validate_spike_array, validate_state_vector

Float64Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass
class AdExState:
    V_mV: Float64Array  # shape (N,)
    w_pA: Float64Array  # shape (N,)
    spiked: BoolArray  # shape (N,)


@dataclass(frozen=True)
class IntegrationMetrics:
    lte_estimate: float
    global_error_bound: float
    recommended_dt_ms: float


def adex_step(
    state: AdExState,
    params: AdExParams,
    dt_ms: float,
    I_syn_pA: Float64Array,
    I_ext_pA: Float64Array,
) -> AdExState:
    """One explicit Euler step for AdEx with spike-reset and exponential clamp.

    Equations follow Brette & Gerstner (2005) with standard reset:
      - if V > Vpeak: V <- Vreset, w <- w + b
    """
    if dt_ms <= 0:
        raise ValueError("dt_ms must be positive")
    N = state.V_mV.shape[0]
    validate_state_vector(state.V_mV, N, name="V_mV")
    validate_state_vector(state.w_pA, N, name="w_pA")
    validate_spike_array(state.spiked, N, name="spiked")
    validate_state_vector(I_syn_pA, N, name="I_syn_pA")
    validate_state_vector(I_ext_pA, N, name="I_ext_pA")

    V = np.asarray(state.V_mV, dtype=np.float64).copy()
    w = np.asarray(state.w_pA, dtype=np.float64).copy()

    # Convert conductances/currents: parameters are in pF/nS/mV/ms/pA so ms is consistent.
    # dV/dt = ( -gL(V-EL) + gL*DeltaT*exp((V-VT)/DeltaT) - w - I_syn + I_ext ) / C
    exp_arg = (V - params.VT_mV) / params.DeltaT_mV
    exp_arg = np.minimum(exp_arg, 20.0)  # prevent overflow
    I_exp = params.gL_nS * params.DeltaT_mV * np.exp(exp_arg)  # nS*mV ~ pA
    dV = (-params.gL_nS * (V - params.EL_mV) + I_exp - w - I_syn_pA + I_ext_pA) / params.C_pF
    V = V + dt_ms * dV

    # dw/dt = ( a(V-EL) - w ) / tauw
    dw = (params.a_nS * (V - params.EL_mV) - w) / params.tauw_ms
    w = w + dt_ms * dw

    spiked = np.asarray(V >= params.Vpeak_mV, dtype=np.bool_)
    if np.any(spiked):
        V[spiked] = params.Vreset_mV
        w[spiked] = w[spiked] + params.b_pA

    return AdExState(V_mV=V, w_pA=w, spiked=spiked)


def adex_step_with_error_tracking(
    state: AdExState,
    params: AdExParams,
    dt_ms: float,
    I_syn_pA: Float64Array,
    I_ext_pA: Float64Array,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-3,
) -> tuple[AdExState, IntegrationMetrics]:
    """One Euler step with step-doubling error tracking for AdEx dynamics."""
    if dt_ms <= 0:
        raise ValueError("dt_ms must be positive")
    if atol <= 0 or rtol <= 0:
        raise ValueError("atol and rtol must be positive")

    full = adex_step(state, params, dt_ms, I_syn_pA, I_ext_pA)
    half = adex_step(state, params, dt_ms * 0.5, I_syn_pA, I_ext_pA)
    half2 = adex_step(half, params, dt_ms * 0.5, I_syn_pA, I_ext_pA)

    delta_v = np.abs(full.V_mV - half2.V_mV)
    delta_w = np.abs(full.w_pA - half2.w_pA)
    scale_v = atol + rtol * np.maximum(np.abs(full.V_mV), np.abs(half2.V_mV))
    scale_w = atol + rtol * np.maximum(np.abs(full.w_pA), np.abs(half2.w_pA))
    err_v = delta_v / scale_v
    err_w = delta_w / scale_w
    lte_estimate = float(np.max(np.concatenate([err_v, err_w])))
    global_error_bound = lte_estimate

    safety = 0.9
    order = 1.0
    err = max(lte_estimate, 1e-12)
    factor = safety * err ** (-1.0 / (order + 1.0))
    recommended_dt_ms = float(dt_ms * min(2.0, max(0.1, factor)))

    metrics = IntegrationMetrics(
        lte_estimate=lte_estimate,
        global_error_bound=global_error_bound,
        recommended_dt_ms=recommended_dt_ms,
    )
    return full, metrics
