"""Information integration metrics for BN-Syn cold emergence.

Implements IIT-inspired metrics:
- Integrated Information (Φ) - measures system integration
- Synergy vs redundancy analysis
- Mutual information calculations
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import entropy as scipy_entropy


@dataclass
class IntegratedInformationMetric:
    """Computes integrated information (Φ) for cold emergence validation.

    Φ (phi) measures how much the system is integrated as a whole,
    rather than being a sum of independent parts.

    Cold emergence ↔ high Φ + low entropy
    """

    min_phi: float = 0.0

    def compute_phi(
        self,
        state: np.ndarray,
        partition_size: int = 2,
    ) -> float:
        """Compute simplified integrated information.

        Φ = MI(whole) - MI(partitions)

        Cold emergence shows: Φ_cold >> Φ_hot (motivational)

        Args:
            state: System state vector
            partition_size: Size of partitions for comparison

        Returns:
            Integrated information value (Φ)
        """
        if len(state) < partition_size * 2:
            return 0.0

        # Mutual information of whole system
        whole_mi = self._mutual_information(state)

        # Partition system and compute MI
        n_partitions = max(1, len(state) // partition_size)
        partition_mi = 0.0

        for i in range(n_partitions):
            start_idx = i * partition_size
            end_idx = min((i + 1) * partition_size, len(state))
            partition = state[start_idx:end_idx]
            if len(partition) > 0:
                partition_mi += self._mutual_information(partition)

        # Φ is the difference (integration beyond parts)
        phi = float(whole_mi - partition_mi / max(1, n_partitions))
        return max(self.min_phi, phi)

    def _mutual_information(self, substate: np.ndarray) -> float:
        """Compute mutual information using entropy estimation.

        Args:
            substate: Subset of state vector

        Returns:
            Mutual information estimate
        """
        if len(substate) == 0:
            return 0.0

        # Normalize to probability-like values
        normalized = np.abs(substate) + 1e-10
        normalized = normalized / np.sum(normalized)

        # Shannon entropy as proxy for MI
        h = float(scipy_entropy(normalized))
        return h

    def compute_synergy(
        self, state: np.ndarray, partition_indices: list[int] | None = None
    ) -> float:
        """Compute synergy vs redundancy ratio.

        Synergy > 0.5 indicates information is synergistic (emergent)
        Synergy < 0.5 indicates redundancy

        Args:
            state: System state vector
            partition_indices: Optional custom partition boundaries

        Returns:
            Synergy ratio [0, 1]
        """
        if partition_indices is None:
            mid = len(state) // 2
            partition_indices = [mid]

        if len(state) < 2:
            return 0.5

        # Compute entropy of whole and parts
        whole_entropy = self._compute_entropy(state)

        parts_entropy = 0.0
        prev_idx = 0
        for idx in partition_indices:
            if prev_idx < idx <= len(state):
                part = state[prev_idx:idx]
                parts_entropy += self._compute_entropy(part)
                prev_idx = idx

        # Add final partition
        if prev_idx < len(state):
            part = state[prev_idx:]
            parts_entropy += self._compute_entropy(part)

        # Synergy: whole entropy vs sum of parts
        # If whole < parts: synergistic (information loss when separated)
        if parts_entropy < 1e-10:
            return 0.5

        synergy_ratio = 1.0 - (whole_entropy / parts_entropy)
        return float(np.clip(synergy_ratio, 0.0, 1.0))

    def _compute_entropy(self, data: np.ndarray) -> float:
        """Compute Shannon entropy of data.

        Args:
            data: Data vector

        Returns:
            Entropy value
        """
        if len(data) == 0:
            return 0.0

        # Normalize to probability distribution
        normalized = np.abs(data) + 1e-10
        normalized = normalized / np.sum(normalized)

        return float(scipy_entropy(normalized))
