"""Validation helpers for crystallizer configuration and observations."""

from __future__ import annotations


def validate_crystallizer_config(
    state_dim: int,
    max_buffer_size: int,
    snapshot_dim: int,
    pca_update_interval: int,
    cluster_eps: float,
    cluster_min_samples: int,
) -> None:
    """Validate crystallizer constructor parameters."""
    if state_dim <= 0:
        raise ValueError("state_dim must be positive")
    if max_buffer_size <= 0:
        raise ValueError("max_buffer_size must be positive")
    if snapshot_dim <= 0 or snapshot_dim > state_dim:
        raise ValueError(f"snapshot_dim must be in (0, {state_dim}], got {snapshot_dim}")
    if pca_update_interval <= 0:
        raise ValueError("pca_update_interval must be positive")
    if cluster_eps <= 0:
        raise ValueError("cluster_eps must be positive")
    if cluster_min_samples <= 0:
        raise ValueError("cluster_min_samples must be positive")


def validate_observation_temperature(temperature: float) -> None:
    """Validate observed temperature value."""
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
