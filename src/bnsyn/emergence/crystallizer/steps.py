"""Numerical step helpers for crystallizer internals."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .types import Attractor, CrystallizerRuntimeState, Float64Array, Phase

logger = logging.getLogger(__name__)


def subsample_state(state: Float64Array, state_dim: int, snapshot_dim: int) -> Float64Array:
    """Subsample a full state vector to snapshot dimensionality."""
    if state.shape[0] != state_dim:
        raise ValueError(f"Expected state shape ({state_dim},), got {state.shape}")
    if state_dim == snapshot_dim:
        return state.copy()
    indices = np.linspace(0, state_dim - 1, snapshot_dim, dtype=int)
    return state[indices]


def current_buffer(state: CrystallizerRuntimeState) -> Float64Array:
    """Return a copy of currently valid buffer rows."""
    if state.buffer_filled:
        return state.buffer.copy()
    return state.buffer[: state.buffer_idx].copy()


def update_pca(state: CrystallizerRuntimeState) -> None:
    """Update PCA basis from the currently buffered data."""
    data = current_buffer(state)
    if data.shape[0] < 2:
        return
    mean = np.mean(data, axis=0)
    centered = data - mean
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        state.pca_components = vt
        state.pca_mean = mean
    except np.linalg.LinAlgError as exc:
        logger.warning(
            "PCA SVD failed (data shape=%s): %s. Retaining previous components.",
            data.shape,
            exc,
        )


def transform_to_pca(snapshot: Float64Array, state: CrystallizerRuntimeState) -> Float64Array:
    """Project one snapshot into PCA coordinates if PCA is available."""
    if state.pca_components is None or state.pca_mean is None:
        return snapshot
    centered = snapshot - state.pca_mean
    return centered @ state.pca_components.T


def dbscan_lite(data: Float64Array, cluster_eps: float, cluster_min_samples: int) -> list[list[int]]:
    """Cluster states with KDTree-based lightweight DBSCAN behavior."""
    if data.shape[0] < cluster_min_samples:
        return []
    tree = cKDTree(data)
    visited = np.zeros(data.shape[0], dtype=bool)
    clusters: list[list[int]] = []
    for i in range(data.shape[0]):
        if visited[i]:
            continue
        neighbors = tree.query_ball_point(data[i], cluster_eps)
        if len(neighbors) < cluster_min_samples:
            visited[i] = True
            continue
        cluster: list[int] = []
        to_visit = list(neighbors)
        while to_visit:
            idx = to_visit.pop(0)
            if visited[idx]:
                continue
            visited[idx] = True
            cluster.append(idx)
            idx_neighbors = tree.query_ball_point(data[idx], cluster_eps)
            if len(idx_neighbors) >= cluster_min_samples:
                to_visit.extend(idx_neighbors)
        if len(cluster) >= cluster_min_samples:
            clusters.append(cluster)
    return clusters


def detect_attractors(state: CrystallizerRuntimeState, cluster_eps: float, cluster_min_samples: int) -> list[Attractor]:
    """Detect attractors from buffered observations."""
    buffer_data = current_buffer(state)
    if buffer_data.shape[0] < cluster_min_samples:
        return []
    if state.pca_components is not None and state.pca_mean is not None:
        pca_data = np.array([transform_to_pca(s, state) for s in buffer_data])
    else:
        pca_data = buffer_data
    clusters = dbscan_lite(pca_data, cluster_eps, cluster_min_samples)
    attractors: list[Attractor] = []
    for cluster_indices in clusters:
        cluster_points = buffer_data[cluster_indices]
        center = np.mean(cluster_points, axis=0)
        distances = np.linalg.norm(cluster_points - center, axis=1)
        stability = float(len(cluster_indices) / buffer_data.shape[0])
        attractors.append(
            Attractor(
                center=center,
                basin_radius=float(np.max(distances)),
                stability=stability,
                formation_step=state.observation_count,
                crystallization=min(1.0, stability * 2.0),
            )
        )
    return attractors


def update_phase(attractors: list[Attractor], current_phase: Phase) -> Phase:
    """Infer crystallization phase from attractor set stability."""
    num_attractors = len(attractors)
    if num_attractors == 0:
        return Phase.FLUID
    if num_attractors == 1:
        return Phase.NUCLEATION
    if num_attractors <= 3:
        return Phase.GROWTH
    max_stability = max(a.stability for a in attractors)
    if max_stability > 0.8:
        return Phase.CRYSTALLIZED
    return Phase.GROWTH


def crystallization_progress(attractors: list[Attractor]) -> float:
    """Compute bounded progress scalar from attractor stability."""
    if not attractors:
        return 0.0
    total_stability = sum(a.stability for a in attractors)
    return float(min(1.0, total_stability))


def dominant_attractor_index(attractors: list[Attractor]) -> int | None:
    """Return index of most stable attractor when available."""
    if not attractors:
        return None
    stabilities: NDArray[np.float64] = np.array([a.stability for a in attractors], dtype=np.float64)
    return int(np.argmax(stabilities))
