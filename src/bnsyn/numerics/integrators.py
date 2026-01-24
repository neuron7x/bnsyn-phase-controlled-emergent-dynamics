"""Deterministic numerical integration utilities.

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
SPEC.md §P2-8

Claims
------
CLM-0022
"""

from __future__ import annotations

import math
from typing import Callable, TypeVar

import numpy as np

State = TypeVar("State")


def clamp_exp_arg(x: float, max_arg: float = 20.0) -> float:
    """Clamp exponential arguments to a maximum value.

    Parameters
    ----------
    x : float
        Raw exponential argument.
    max_arg : float
        Maximum allowed argument value.

    Returns
    -------
    float
        Clamped argument.

    Determinism
    -----------
    Deterministic under fixed inputs.

    SPEC
    ----
    SPEC.md §P2-8

    Claims
    ------
    CLM-0022
    """
    return float(min(x, max_arg))


def euler_step(x: np.ndarray, dt: float, f: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Advance one explicit Euler step.

    Parameters
    ----------
    x : numpy.ndarray
        Current state vector.
    dt : float
        Timestep.
    f : Callable[[numpy.ndarray], numpy.ndarray]
        Right-hand side function.

    Returns
    -------
    numpy.ndarray
        Updated state vector.

    Determinism
    -----------
    Deterministic under fixed inputs.

    SPEC
    ----
    SPEC.md §P2-8

    Claims
    ------
    CLM-0022
    """
    return x + dt * f(x)


def rk2_step(x: np.ndarray, dt: float, f: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Advance one second-order Runge-Kutta step.

    Parameters
    ----------
    x : numpy.ndarray
        Current state vector.
    dt : float
        Timestep.
    f : Callable[[numpy.ndarray], numpy.ndarray]
        Right-hand side function.

    Returns
    -------
    numpy.ndarray
        Updated state vector.

    Determinism
    -----------
    Deterministic under fixed inputs.

    SPEC
    ----
    SPEC.md §P2-8

    Claims
    ------
    CLM-0022
    """
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    return x + dt * k2


def exp_decay_step(g: np.ndarray, dt: float, tau: float) -> np.ndarray:
    """Unconditionally stable exponential decay: g <- g * exp(-dt/tau).

    Parameters
    ----------
    g : numpy.ndarray
        Current value.
    dt : float
        Timestep.
    tau : float
        Time constant.

    Returns
    -------
    numpy.ndarray
        Updated value after exponential decay.

    Determinism
    -----------
    Deterministic under fixed inputs.

    SPEC
    ----
    SPEC.md §P2-8

    Claims
    ------
    CLM-0022
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    return g * math.exp(-dt / tau)
