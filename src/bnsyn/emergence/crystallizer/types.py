"""Typed data structures for attractor crystallization tracking."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

Float64Array = NDArray[np.float64]


class Phase(Enum):
    """Crystallization phase labels."""

    FLUID = "fluid"
    NUCLEATION = "nucleation"
    GROWTH = "growth"
    CRYSTALLIZED = "crystallized"


@dataclass(frozen=True)
class Attractor:
    """Detected attractor descriptor."""

    center: Float64Array
    basin_radius: float
    stability: float
    formation_step: int
    crystallization: float


@dataclass(frozen=True)
class CrystallizationState:
    """Snapshot of global crystallization state."""

    progress: float
    num_attractors: int
    dominant_attractor: int | None
    phase: Phase
    temperature: float


@dataclass
class CrystallizerRuntimeState:
    """Mutable runtime storage for crystallizer internals."""

    buffer: Float64Array
    buffer_idx: int = 0
    buffer_filled: bool = False
    observation_count: int = 0
    pca_components: Float64Array | None = None
    pca_mean: Float64Array | None = None
    attractors: list[Attractor] | None = None
    current_phase: Phase = Phase.FLUID
    current_temperature: float = 1.0

    def __post_init__(self) -> None:
        """Initialize optional mutable fields."""
        if self.attractors is None:
            self.attractors = []
