"""Three-factor plasticity rules for synaptic updates.

Parameters
----------
None

Returns
-------
None

Determinism
-----------
Deterministic under fixed inputs and fixed timestep.

SPEC
----
SPEC.md §P0-3

Claims
------
CLM-0004, CLM-0005
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from bnsyn.config import PlasticityParams

Float64Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass
class EligibilityTraces:
    """Store eligibility traces for synapses.

    Parameters
    ----------
    e : numpy.ndarray
        Eligibility matrix (shape: [N_pre, N_post]).

    Returns
    -------
    EligibilityTraces
        Eligibility trace container.

    Determinism
    -----------
    Deterministic given fixed inputs.

    SPEC
    ----
    SPEC.md §P0-3

    Claims
    ------
    CLM-0004
    """

    e: Float64Array  # shape (N_pre, N_post)


@dataclass
class NeuromodulatorTrace:
    """Store neuromodulator trace value.

    Parameters
    ----------
    n : float
        Scalar neuromodulator value (dimensionless).

    Returns
    -------
    NeuromodulatorTrace
        Neuromodulator trace container.

    Determinism
    -----------
    Deterministic given fixed inputs.

    SPEC
    ----
    SPEC.md §P0-3

    Claims
    ------
    CLM-0005
    """

    n: float  # scalar dopamine / TD trace


def decay(x: Float64Array, dt_ms: float, tau_ms: float) -> Float64Array:
    """Apply exponential decay to a trace.

    Parameters
    ----------
    x : numpy.ndarray
        Trace array to decay.
    dt_ms : float
        Timestep in milliseconds.
    tau_ms : float
        Time constant in milliseconds.

    Returns
    -------
    numpy.ndarray
        Decayed trace array.

    Determinism
    -----------
    Deterministic under fixed inputs.

    SPEC
    ----
    SPEC.md §P0-3

    Claims
    ------
    CLM-0004
    """
    return np.asarray(x * np.exp(-dt_ms / tau_ms), dtype=np.float64)


def three_factor_update(
    w: Float64Array,
    elig: EligibilityTraces,
    neuromod: NeuromodulatorTrace,
    pre_spikes: BoolArray,
    post_spikes: BoolArray,
    dt_ms: float,
    p: PlasticityParams,
) -> tuple[Float64Array, EligibilityTraces]:
    """Update synaptic weights using three-factor learning.

    Parameters
    ----------
    w : numpy.ndarray
        Weight matrix (shape: [N_pre, N_post]).
    elig : EligibilityTraces
        Eligibility trace container.
    neuromod : NeuromodulatorTrace
        Neuromodulator trace.
    pre_spikes : numpy.ndarray
        Presynaptic spike indicators (shape: [N_pre]).
    post_spikes : numpy.ndarray
        Postsynaptic spike indicators (shape: [N_post]).
    dt_ms : float
        Timestep in milliseconds (must be positive).
    p : PlasticityParams
        Plasticity parameter set.

    Returns
    -------
    tuple[numpy.ndarray, EligibilityTraces]
        Updated weights and eligibility traces.

    Determinism
    -----------
    Deterministic under fixed inputs.

    SPEC
    ----
    SPEC.md §P0-3

    Claims
    ------
    CLM-0004, CLM-0005
    """
    if w.ndim != 2:
        raise ValueError("w must be 2D (N_pre, N_post)")
    if elig.e.shape != w.shape:
        raise ValueError("eligibility shape must match weights")
    if pre_spikes.shape != (w.shape[0],):
        raise ValueError("pre_spikes shape mismatch")
    if post_spikes.shape != (w.shape[1],):
        raise ValueError("post_spikes shape mismatch")
    if dt_ms <= 0:
        raise ValueError("dt_ms must be positive")

    e = decay(elig.e, dt_ms, p.tau_e_ms)

    # coincidence term: for spikes at this dt, add fixed increments approximating STDP window
    # (for full spike-time STDP, track last spike times; see docs/SPEC.md)
    coincidence = np.outer(
        pre_spikes.astype(np.float64, copy=False), post_spikes.astype(np.float64, copy=False)
    )
    e = e + coincidence

    dw = p.eta * e * float(neuromod.n)
    w_new = np.asarray(np.clip(w + dw, p.w_min, p.w_max), dtype=np.float64)

    return w_new, EligibilityTraces(e=e)


def neuromod_step(n: float, dt_ms: float, tau_ms: float, d_t: float) -> float:
    """Update neuromodulator trace with exponential decay and drive.

    Parameters
    ----------
    n : float
        Current neuromodulator value.
    dt_ms : float
        Timestep in milliseconds.
    tau_ms : float
        Time constant in milliseconds.
    d_t : float
        Driving term for neuromodulation at this step.

    Returns
    -------
    float
        Updated neuromodulator value.

    Determinism
    -----------
    Deterministic under fixed inputs.

    SPEC
    ----
    SPEC.md §P0-3

    Claims
    ------
    CLM-0005
    """
    n = float(n) * float(np.exp(-dt_ms / tau_ms)) + float(d_t)
    return float(n)
