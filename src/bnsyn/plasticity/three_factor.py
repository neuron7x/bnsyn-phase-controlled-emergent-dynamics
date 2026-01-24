"""Three-factor plasticity updates with eligibility and neuromodulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from bnsyn.config import PlasticityParams

Float64Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass
class EligibilityTraces:
    """Eligibility traces for synaptic updates."""

    e: Float64Array  # shape (N_pre, N_post)


@dataclass
class NeuromodulatorTrace:
    """Neuromodulator trace for global reward signals."""

    n: float  # scalar dopamine / TD trace


def decay(x: Float64Array, dt_ms: float, tau_ms: float) -> Float64Array:
    """Exponentially decay a trace.

    Parameters
    ----------
    x
        Trace values.
    dt_ms
        Time step in milliseconds.
    tau_ms
        Time constant in milliseconds.

    Returns
    -------
    numpy.ndarray
        Decayed trace values.
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
    """Apply a vectorized three-factor plasticity update.

    Parameters
    ----------
    w
        Current weight matrix.
    elig
        Eligibility traces.
    neuromod
        Neuromodulator trace.
    pre_spikes
        Pre-synaptic spike indicator array.
    post_spikes
        Post-synaptic spike indicator array.
    dt_ms
        Time step in milliseconds.
    p
        Plasticity parameters.

    Returns
    -------
    tuple[numpy.ndarray, EligibilityTraces]
        Updated weights and eligibility traces.

    Raises
    ------
    ValueError
        If shapes mismatch or ``dt_ms`` is non-positive.
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
    """Update neuromodulator trace by exponential decay plus drive.

    Parameters
    ----------
    n
        Current neuromodulator level.
    dt_ms
        Time step in milliseconds.
    tau_ms
        Decay constant in milliseconds.
    d_t
        Instantaneous drive term.

    Returns
    -------
    float
        Updated neuromodulator level.
    """
    n = float(n) * float(np.exp(-dt_ms / tau_ms)) + float(d_t)
    return float(n)
