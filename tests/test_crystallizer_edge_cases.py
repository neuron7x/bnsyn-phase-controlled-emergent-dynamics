import numpy as np
import pytest

from bnsyn.emergence.crystallizer import AttractorCrystallizer


def test_crystallizer_invalid_params() -> None:
    with pytest.raises(ValueError, match="state_dim must be positive"):
        AttractorCrystallizer(state_dim=0)
    with pytest.raises(ValueError, match="max_buffer_size must be positive"):
        AttractorCrystallizer(state_dim=2, max_buffer_size=0)
    with pytest.raises(ValueError, match="snapshot_dim must be in"):
        AttractorCrystallizer(state_dim=2, snapshot_dim=3)
    with pytest.raises(ValueError, match="pca_update_interval must be positive"):
        AttractorCrystallizer(state_dim=2, snapshot_dim=2, pca_update_interval=0)
    with pytest.raises(ValueError, match="cluster_eps must be positive"):
        AttractorCrystallizer(state_dim=2, snapshot_dim=2, cluster_eps=0.0)
    with pytest.raises(ValueError, match="cluster_min_samples must be positive"):
        AttractorCrystallizer(state_dim=2, snapshot_dim=2, cluster_min_samples=0)


def test_crystallizer_subsample_shape_validation() -> None:
    crystallizer = AttractorCrystallizer(state_dim=4, snapshot_dim=2)
    with pytest.raises(ValueError, match="Expected state shape"):
        crystallizer._subsample_state(np.zeros(3))


def test_crystallizer_transform_default_passthrough() -> None:
    crystallizer = AttractorCrystallizer(state_dim=2, snapshot_dim=2)
    state = np.array([0.1, 0.2], dtype=np.float64)
    transformed = crystallizer._transform_to_pca(state)
    assert np.allclose(transformed, state)


def test_crystallizer_pca_failure_retains_previous(monkeypatch: pytest.MonkeyPatch) -> None:
    crystallizer = AttractorCrystallizer(state_dim=2, snapshot_dim=2, max_buffer_size=5)
    crystallizer._buffer[:2] = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    crystallizer._buffer_idx = 2

    def _raise_svd(*_args: object, **_kwargs: object) -> None:
        raise np.linalg.LinAlgError("svd failure")

    monkeypatch.setattr(np.linalg, "svd", _raise_svd)
    crystallizer._update_pca()


def test_crystallizer_dbscan_detects_cluster() -> None:
    crystallizer = AttractorCrystallizer(
        state_dim=2, snapshot_dim=2, cluster_eps=0.2, cluster_min_samples=2
    )
    data = np.array([[0.0, 0.0], [0.05, 0.05], [1.0, 1.0]], dtype=np.float64)
    clusters = crystallizer._dbscan_lite(data)
    assert any(len(cluster) >= 2 for cluster in clusters)
