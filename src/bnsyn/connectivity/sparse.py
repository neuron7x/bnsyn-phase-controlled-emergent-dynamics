"""Sparse connectivity utilities with CSR representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

Float64Array = NDArray[np.float64]


@dataclass(frozen=True)
class SparseConnectivityMetrics:
    """Sparsity metrics and performance estimates."""

    density: float
    sparsity: float
    nnz: int
    memory_dense_mb: float
    memory_sparse_mb: float
    speedup_estimated: float


class SparseConnectivity:
    """Adaptive sparse/dense matrix dispatcher."""

    def __init__(
        self,
        W: Float64Array,
        density_threshold: float = 0.10,
        force_format: Literal["auto", "dense", "sparse"] = "auto",
    ) -> None:
        if W.ndim != 2:
            raise ValueError("W must be 2D")
        if W.dtype != np.float64:
            W = np.asarray(W, dtype=np.float64)

        self.shape = W.shape
        n_pre, n_post = W.shape
        nnz = int(np.count_nonzero(W))
        density = nnz / (n_pre * n_post) if (n_pre * n_post) > 0 else 0.0

        if force_format == "auto":
            self.format: Literal["dense", "sparse"] = (
                "sparse" if density < density_threshold else "dense"
            )
        else:
            self.format = force_format

        if self.format == "sparse":
            self.W: sp.csr_matrix | Float64Array = sp.csr_matrix(W, dtype=np.float64)
        else:
            self.W = W.astype(np.float64, copy=False)

        self.metrics = SparseConnectivityMetrics(
            density=density,
            sparsity=1.0 - density,
            nnz=nnz,
            memory_dense_mb=(n_pre * n_post * 8) / (1024**2),
            memory_sparse_mb=self._estimate_sparse_size(n_pre, nnz),
            speedup_estimated=self._estimate_speedup(density),
        )

    @staticmethod
    def _estimate_sparse_size(n_pre: int, nnz: int) -> float:
        bytes_used = nnz * 8 + nnz * 4 + (n_pre + 1) * 4
        return bytes_used / (1024**2)

    @staticmethod
    def _estimate_speedup(density: float) -> float:
        if density >= 0.1:
            return 1.0
        return min(10.0, 2.0 ** (1.0 - density) / density) if density > 0 else 10.0

    def apply(self, x: Float64Array) -> Float64Array:
        """Compute y = W @ x with automatic dispatch."""
        if self.format == "sparse":
            assert isinstance(self.W, sp.csr_matrix)
            y = self.W @ x
            if sp.issparse(y):
                return np.asarray(y.todense(), dtype=np.float64).ravel()
            return np.asarray(y, dtype=np.float64).ravel()
        return np.asarray(np.dot(self.W, x), dtype=np.float64)

    def to_dense(self) -> Float64Array:
        """Convert to dense matrix."""
        if self.format == "sparse":
            assert isinstance(self.W, sp.csr_matrix)
            return np.asarray(self.W.todense(), dtype=np.float64)
        return np.asarray(self.W, dtype=np.float64)

    def to_sparse(self) -> sp.csr_matrix:
        """Convert to sparse CSR."""
        if self.format == "sparse":
            return self.W.copy()
        return sp.csr_matrix(self.W, dtype=np.float64)

    def __repr__(self) -> str:
        return (
            f"SparseConnectivity(shape={self.shape}, "
            f"format={self.format}, density={self.metrics.density:.1%}, "
            f"memory={self.metrics.memory_sparse_mb:.2f}MB)"
        )


def build_random_connectivity(
    n_pre: int,
    n_post: int,
    connection_prob: float,
    weight_mean: float = 1.0,
    weight_std: float = 0.1,
    seed: int | None = None,
) -> SparseConnectivity:
    """Build Erdős-Rényi random connectivity."""
    if n_pre <= 0 or n_post <= 0:
        raise ValueError("n_pre and n_post must be positive")
    if not (0.0 <= connection_prob <= 1.0):
        raise ValueError("connection_prob must be in [0,1]")
    rng = np.random.default_rng(seed)
    is_connected = rng.binomial(1, connection_prob, (n_pre, n_post))
    weights = np.abs(rng.normal(weight_mean, weight_std, (n_pre, n_post)))
    W = np.asarray(is_connected * weights, dtype=np.float64)
    return SparseConnectivity(W)
