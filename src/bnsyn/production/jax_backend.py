"""Optional JAX backend experiments.

Parameters
----------
None

Returns
-------
None

Determinism
-----------
Deterministic under fixed inputs and fixed parameters.

SPEC
----
SPEC.md §P0-1

Claims
------
None
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
    """Advance AdEx dynamics by one step using JAX arrays.

    Parameters
    ----------
    V : Any
        Membrane potentials.
    w : Any
        Adaptation currents.
    input_current : Any
        Input currents.
    C : float
        Membrane capacitance (F).
    gL : float
        Leak conductance (S).
    EL : float
        Leak reversal potential (V).
    VT : float
        Threshold voltage (V).
    DeltaT : float
        Slope factor (V).
    tau_w : float
        Adaptation time constant (s).
    a : float
        Adaptation conductance (S).
    b : float
        Adaptation increment (A).
    V_reset : float
        Reset voltage (V).
    V_spike : float
        Spike threshold (V).
    dt : float
        Timestep (s).

    Returns
    -------
    tuple[Any, Any, Any]
        Updated V, updated w, and spike indicators.

    Determinism
    -----------
    Deterministic under fixed inputs.

    SPEC
    ----
    SPEC.md §P0-1

    Claims
    ------
    None
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
