"""Optional JAX backend experiments.

This module is intentionally optional. It is not imported by default.
"""

from __future__ import annotations

try:
    import jax.numpy as jnp
except Exception as e:  # pragma: no cover
    raise ImportError(
        "JAX is not installed. Install an appropriate jax/jaxlib build to use this module."
    ) from e


def adex_step_jax(V, w, I, *, C, gL, EL, VT, DeltaT, tau_w, a, b, V_reset, V_spike, dt):
    exp_term = gL * DeltaT * jnp.exp((V - VT) / DeltaT)
    dV = (-(gL * (V - EL)) + exp_term - w + I) / C
    dw = (a * (V - EL) - w) / tau_w
    V_new = V + dt * dV
    w_new = w + dt * dw
    spikes = V_new >= V_spike
    V_new = jnp.where(spikes, V_reset, V_new)
    w_new = jnp.where(spikes, w_new + b, w_new)
    return V_new, w_new, spikes
