"""Pydantic validation models and NumPy input validators."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from bnsyn.config import AdExParams, CriticalityParams, SynapseParams

Float64Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


class NetworkValidationConfig(BaseModel):
    """Validated network configuration for API boundaries."""

    model_config = ConfigDict(frozen=True, validate_assignment=True)

    N: int = Field(default=200, gt=0, le=100_000)
    frac_inhib: float = Field(default=0.2, gt=0.0, lt=1.0)
    p_conn: float = Field(default=0.05, gt=0.0, lt=1.0)
    w_exc_nS: float = Field(default=0.5, gt=0.0)
    w_inh_nS: float = Field(default=1.0, gt=0.0)
    ext_rate_hz: float = Field(default=2.0, ge=0.0)
    ext_w_nS: float = Field(default=0.3, ge=0.0)
    dt_ms: float = Field(default=0.1, gt=0.0, le=1.0)

    adex: AdExParams = Field(default_factory=AdExParams)
    syn: SynapseParams = Field(default_factory=SynapseParams)
    crit: CriticalityParams = Field(default_factory=CriticalityParams)


def _ensure_ndarray(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name}: expected ndarray, got {type(value)}")
    return value


def validate_state_vector(state: Float64Array, n_neurons: int, name: str = "state") -> None:
    """Validate a 1D float64 state vector."""
    arr = _ensure_ndarray(state, name)
    if arr.dtype != np.float64:
        raise ValueError(f"{name}: expected dtype float64, got {arr.dtype}")
    if arr.shape != (n_neurons,):
        raise ValueError(f"{name}: expected shape ({n_neurons},), got {arr.shape}")
    if np.any(np.isnan(arr)):
        raise ValueError(f"{name}: contains NaN")


def validate_spike_array(spikes: BoolArray, n_neurons: int, name: str = "spikes") -> None:
    """Validate a 1D boolean spike array."""
    arr = _ensure_ndarray(spikes, name)
    if arr.dtype != np.bool_:
        raise ValueError(f"{name}: expected dtype bool, got {arr.dtype}")
    if arr.shape != (n_neurons,):
        raise ValueError(f"{name}: expected shape ({n_neurons},), got {arr.shape}")


def validate_connectivity_matrix(
    matrix: Float64Array,
    shape: tuple[int, int],
    name: str = "connectivity",
) -> None:
    """Validate a 2D float64 connectivity matrix."""
    arr = _ensure_ndarray(matrix, name)
    if arr.dtype != np.float64:
        raise ValueError(f"{name}: expected dtype float64, got {arr.dtype}")
    if arr.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {arr.shape}")
    if np.any(np.isnan(arr)):
        raise ValueError(f"{name}: contains NaN")
