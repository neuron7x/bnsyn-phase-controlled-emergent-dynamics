from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from bnsyn.config import PlasticityParams


@dataclass
class EligibilityTraces:
    e: np.ndarray  # shape (N_pre, N_post)


@dataclass
class NeuromodulatorTrace:
    n: float  # scalar dopamine / TD trace


def decay(x: np.ndarray, dt_ms: float, tau_ms: float) -> np.ndarray:
    return cast(np.ndarray, x * np.exp(-dt_ms / tau_ms))


def three_factor_update(
    w: np.ndarray,
    elig: EligibilityTraces,
    neuromod: NeuromodulatorTrace,
    pre_spikes: np.ndarray,
    post_spikes: np.ndarray,
    dt_ms: float,
    p: PlasticityParams,
) -> tuple[np.ndarray, EligibilityTraces]:
    """Vectorized three-factor update.

    - eligibility trace receives STDP-like coincidence: outer(pre, post)
    - weights updated by eta * e * M
    - weights bounded to [w_min, w_max]
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
    coincidence = np.outer(pre_spikes.astype(float), post_spikes.astype(float))
    e = e + coincidence

    dw = p.eta * e * float(neuromod.n)
    w_new = np.clip(w + dw, p.w_min, p.w_max)

    return w_new, EligibilityTraces(e=e)


def neuromod_step(n: float, dt_ms: float, tau_ms: float, d_t: float) -> float:
    """dn/dt = -n/tau + d(t) (Euler exact via exp)."""
    n = float(n) * float(np.exp(-dt_ms / tau_ms)) + float(d_t)
    return float(n)
