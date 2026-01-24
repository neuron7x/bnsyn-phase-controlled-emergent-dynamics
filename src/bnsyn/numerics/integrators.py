from __future__ import annotations

import math
from typing import Callable, TypeVar

import numpy as np

State = TypeVar("State")


def clamp_exp_arg(x: float, max_arg: float = 20.0) -> float:
    return float(min(x, max_arg))


def euler_step(x: np.ndarray, dt: float, f: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    return x + dt * f(x)


def rk2_step(x: np.ndarray, dt: float, f: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    return x + dt * k2


def exp_decay_step(g: np.ndarray, dt: float, tau: float) -> np.ndarray:
    """Unconditionally stable exponential decay: g <- g * exp(-dt/tau)."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    return g * math.exp(-dt / tau)
