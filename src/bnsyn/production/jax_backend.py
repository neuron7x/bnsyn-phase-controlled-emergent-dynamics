"""Optional JAX backend experiments.

This module is intentionally optional. It is not imported by default.
"""

from __future__ import annotations

from typing import Any

try:
    import jax.numpy as jnp
except Exception as e:  # pragma: no cover
    raise ImportError(
        "JAX is not installed. Install an appropriate jax/jaxlib build to use this module."
    ) from e


def adex_step_jax(
    V: Any,
    w: Any,
    input_current: Any,
    *,
    C: float,
    gL: float,
    EL: float,
    VT: float,
    DeltaT: float,
    tau_w: float,
    a: float,
    b: float,
    V_reset: float,
    V_spike: float,
    dt: float,
) -> tuple[Any, Any, Any]:
    """JAX-accelerated AdEx neuron step.

    Args:
        V: Membrane potential (mV) array.
        w: Adaptation current (pA) array.
        input_current: External input current (pA) array.
        C: Membrane capacitance (pF).
        gL: Leak conductance (nS).
        EL: Leak reversal potential (mV).
        VT: Threshold potential (mV).
        DeltaT: Slope factor (mV).
        tau_w: Adaptation time constant (ms).
        a: Subthreshold adaptation (nS).
        b: Spike-triggered adaptation (pA).
        V_reset: Reset potential after spike (mV).
        V_spike: Spike detection threshold (mV).
        dt: Timestep (ms).

    Returns:
        Tuple of (V_new, w_new, spikes) with updated state and spike mask.
    """
    exp_term = gL * DeltaT * jnp.exp((V - VT) / DeltaT)
    dV = (-(gL * (V - EL)) + exp_term - w + input_current) / C
    dw = (a * (V - EL) - w) / tau_w
    V_new = V + dt * dV
    w_new = w + dt * dw
    spikes = V_new >= V_spike
    V_new = jnp.where(spikes, V_reset, V_new)
    w_new = jnp.where(spikes, w_new + b, w_new)
    return V_new, w_new, spikes
