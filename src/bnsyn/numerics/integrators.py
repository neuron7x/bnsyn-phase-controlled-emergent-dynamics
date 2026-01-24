"""Numerical integration helpers for deterministic dynamics."""

from __future__ import annotations

import math
from typing import Callable, TypeVar

import numpy as np

State = TypeVar("State")


def clamp_exp_arg(x: float, max_arg: float = 20.0) -> float:
    """Clamp an exponential argument to avoid overflow.

    Parameters
    ----------
    x
        Input value.
    max_arg
        Maximum allowed value.

    Returns
    -------
    float
        Clamped value.
    """
    return float(min(x, max_arg))


def euler_step(x: np.ndarray, dt: float, f: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Advance state by one Euler step.

    Parameters
    ----------
    x
        Current state vector.
    dt
        Time step.
    f
        Vector field function ``f(x)``.

    Returns
    -------
    numpy.ndarray
        Updated state.
    """
    return x + dt * f(x)


def rk2_step(x: np.ndarray, dt: float, f: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Advance state by one midpoint RK2 step.

    Parameters
    ----------
    x
        Current state vector.
    dt
        Time step.
    f
        Vector field function ``f(x)``.

    Returns
    -------
    numpy.ndarray
        Updated state.
    """
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    return x + dt * k2


def exp_decay_step(g: np.ndarray, dt: float, tau: float) -> np.ndarray:
    """Apply an exponential decay step ``g <- g * exp(-dt/tau)``.

    Parameters
    ----------
    g
        State to decay.
    dt
        Time step.
    tau
        Time constant.

    Returns
    -------
    numpy.ndarray
        Decayed state.

    Raises
    ------
    ValueError
        If ``tau`` is non-positive.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    return g * math.exp(-dt / tau)
