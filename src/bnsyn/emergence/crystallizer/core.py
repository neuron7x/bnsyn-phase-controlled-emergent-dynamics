"""Crystallizer orchestration logic."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .steps import (
    crystallization_progress,
    dbscan_lite,
    detect_attractors,
    dominant_attractor_index,
    subsample_state,
    transform_to_pca,
    update_pca,
    update_phase,
)
from .types import Attractor, CrystallizationState, CrystallizerRuntimeState, Float64Array, Phase
from .validate import validate_crystallizer_config, validate_observation_temperature


@dataclass
class AttractorCrystallizer:
    """Orchestrate attractor detection and crystallization tracking."""

    state_dim: int
    max_buffer_size: int = 1000
    snapshot_dim: int = 100
    pca_update_interval: int = 100
    cluster_eps: float = 0.1
    cluster_min_samples: int = 5

    _runtime: CrystallizerRuntimeState = field(init=False, repr=False)
    _attractor_callbacks: list[Callable[[Attractor], None]] = field(default_factory=list, init=False, repr=False)
    _phase_callbacks: list[Callable[[Phase, Phase], None]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize validated runtime state storage."""
        validate_crystallizer_config(
            self.state_dim,
            self.max_buffer_size,
            self.snapshot_dim,
            self.pca_update_interval,
            self.cluster_eps,
            self.cluster_min_samples,
        )
        self._runtime = CrystallizerRuntimeState(
            buffer=np.zeros((self.max_buffer_size, self.snapshot_dim), dtype=np.float64)
        )

    @property
    def _buffer(self) -> Float64Array:
        return self._runtime.buffer

    @_buffer.setter
    def _buffer(self, value: Float64Array) -> None:
        self._runtime.buffer = value

    @property
    def _buffer_idx(self) -> int:
        return self._runtime.buffer_idx

    @_buffer_idx.setter
    def _buffer_idx(self, value: int) -> None:
        self._runtime.buffer_idx = value

    @property
    def _buffer_filled(self) -> bool:
        return self._runtime.buffer_filled

    @_buffer_filled.setter
    def _buffer_filled(self, value: bool) -> None:
        self._runtime.buffer_filled = value

    @property
    def _observation_count(self) -> int:
        return self._runtime.observation_count

    @_observation_count.setter
    def _observation_count(self, value: int) -> None:
        self._runtime.observation_count = value

    @property
    def _pca_components(self) -> Float64Array | None:
        return self._runtime.pca_components

    @_pca_components.setter
    def _pca_components(self, value: Float64Array | None) -> None:
        self._runtime.pca_components = value

    @property
    def _pca_mean(self) -> Float64Array | None:
        return self._runtime.pca_mean

    @_pca_mean.setter
    def _pca_mean(self, value: Float64Array | None) -> None:
        self._runtime.pca_mean = value

    @property
    def _attractors(self) -> list[Attractor]:
        return self._runtime.attractors or []

    @_attractors.setter
    def _attractors(self, value: list[Attractor]) -> None:
        self._runtime.attractors = value

    @property
    def _current_phase(self) -> Phase:
        return self._runtime.current_phase

    @_current_phase.setter
    def _current_phase(self, value: Phase) -> None:
        self._runtime.current_phase = value

    @property
    def _current_temperature(self) -> float:
        return self._runtime.current_temperature

    @_current_temperature.setter
    def _current_temperature(self, value: float) -> None:
        self._runtime.current_temperature = value

    def _subsample_state(self, state: Float64Array) -> Float64Array:
        return subsample_state(state, self.state_dim, self.snapshot_dim)

    def _update_pca(self) -> None:
        update_pca(self._runtime)

    def _transform_to_pca(self, state: Float64Array) -> Float64Array:
        return transform_to_pca(state, self._runtime)

    def _dbscan_lite(self, data: Float64Array) -> list[list[int]]:
        return dbscan_lite(data, self.cluster_eps, self.cluster_min_samples)

    def _detect_attractors(self) -> list[Attractor]:
        return detect_attractors(self._runtime, self.cluster_eps, self.cluster_min_samples)

    def _update_phase(self) -> None:
        old_phase = self._runtime.current_phase
        new_phase = update_phase(self._attractors, old_phase)
        self._runtime.current_phase = new_phase
        if old_phase != new_phase:
            for callback in self._phase_callbacks:
                callback(old_phase, new_phase)

    def observe(self, state: Float64Array, temperature: float) -> None:
        """Observe one state sample and update crystallization internals."""
        validate_observation_temperature(temperature)
        self._runtime.current_temperature = temperature
        snapshot = self._subsample_state(state)
        self._runtime.buffer[self._runtime.buffer_idx] = snapshot
        self._runtime.buffer_idx = (self._runtime.buffer_idx + 1) % self.max_buffer_size
        if self._runtime.buffer_idx == 0:
            self._runtime.buffer_filled = True
        self._runtime.observation_count += 1

        should_update = self._runtime.observation_count % self.pca_update_interval == 0
        if not should_update:
            return
        self._update_pca()
        new_attractors = self._detect_attractors()
        for new_attr in new_attractors:
            is_new = all(
                float(np.linalg.norm(new_attr.center - existing.center)) >= self.cluster_eps
                for existing in self._attractors
            )
            if is_new:
                self._attractors.append(new_attr)
                for callback in self._attractor_callbacks:
                    callback(new_attr)
        self._update_phase()

    def get_attractors(self) -> list[Attractor]:
        """Return a copy of known attractors."""
        return self._attractors.copy()

    def crystallization_progress(self) -> float:
        """Return scalar crystallization progress in [0, 1]."""
        return crystallization_progress(self._attractors)

    def get_crystallization_state(self) -> CrystallizationState:
        """Return a crystallization state snapshot."""
        return CrystallizationState(
            progress=self.crystallization_progress(),
            num_attractors=len(self._attractors),
            dominant_attractor=dominant_attractor_index(self._attractors),
            phase=self._runtime.current_phase,
            temperature=self._runtime.current_temperature,
        )

    def on_attractor_formed(self, callback: Callable[[Attractor], None]) -> None:
        """Register callback for newly accepted attractors."""
        self._attractor_callbacks.append(callback)

    def on_phase_transition(self, callback: Callable[[Phase, Phase], None]) -> None:
        """Register callback for phase transition events."""
        self._phase_callbacks.append(callback)
