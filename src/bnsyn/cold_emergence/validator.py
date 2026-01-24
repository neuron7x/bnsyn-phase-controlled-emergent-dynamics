"""Formal validation of cold emergence properties.

Validates that the system exhibits genuine cold emergence through:
- Determinism (negative Lyapunov exponent)
- Integration (high Φ)
- Organization without rewards (synergy > redundancy)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from bnsyn.cold_emergence.information_metrics import IntegratedInformationMetric


@dataclass
class ColdEmergenceValidator:
    """Validates cold emergence axioms.

    Cold emergence axioms:
    1. Determinism: Lyapunov exp < 0 (exponential stability)
    2. Integration: Φ > threshold (non-reducible to parts)
    3. Organization without rewards: synergy > 0.5
    """

    lyapunov_threshold: float = -0.1
    phi_threshold: float = 0.3
    synergy_threshold: float = 0.5

    def validate_cold_emergence(
        self,
        state: np.ndarray,
        lyapunov_exponent: float,
        temperature: float,
    ) -> dict[str, Any]:
        """Comprehensive validation of cold emergence.

        Args:
            state: Current system state
            lyapunov_exponent: Measured Lyapunov exponent
            temperature: Effective system temperature

        Returns:
            Dictionary with validation metrics and overall pass/fail
        """
        # Compute information metrics
        iit = IntegratedInformationMetric()
        phi = iit.compute_phi(state)
        synergy = iit.compute_synergy(state)

        # Validation criteria
        is_deterministic = lyapunov_exponent < self.lyapunov_threshold
        is_integrated = phi > self.phi_threshold
        is_synergistic = synergy > self.synergy_threshold

        # Overall cold emergence validation
        is_cold_emergent = is_deterministic and is_integrated and is_synergistic

        return {
            "lyapunov_exponent": float(lyapunov_exponent),
            "integrated_information_phi": float(phi),
            "synergy_vs_redundancy": float(synergy),
            "temperature_effective": float(temperature),
            "is_deterministic": bool(is_deterministic),
            "is_integrated": bool(is_integrated),
            "is_synergistic": bool(is_synergistic),
            "is_truly_cold_emergent": bool(is_cold_emergent),
        }

    def measure_lyapunov(
        self, state_trajectory: list[np.ndarray], dt: float, eps: float = 1e-8
    ) -> float:
        """Estimate first Lyapunov exponent from trajectory.

        Args:
            state_trajectory: Sequence of system states
            dt: Time step between states
            eps: Small perturbation for numerical stability

        Returns:
            Lyapunov exponent estimate
        """
        if len(state_trajectory) < 2:
            return 0.0

        divergences = []
        for i in range(len(state_trajectory) - 1):
            # Add small perturbation
            perturbed = state_trajectory[i] + eps

            # Measure divergence at next step
            divergence = float(np.linalg.norm(state_trajectory[i + 1] - perturbed))
            if divergence > eps:
                divergences.append(np.log(divergence / eps) / dt)

        if not divergences:
            return 0.0

        return float(np.mean(divergences))

    def measure_integrated_info(self, state: np.ndarray, partition_size: int = 2) -> float:
        """Compute Φ (integrated information).

        Args:
            state: System state
            partition_size: Size of partitions

        Returns:
            Φ value
        """
        iit = IntegratedInformationMetric()
        return iit.compute_phi(state, partition_size)

    def measure_synergy(self, state: np.ndarray) -> float:
        """Compute synergy vs redundancy ratio.

        Args:
            state: System state

        Returns:
            Synergy ratio [0, 1]
        """
        iit = IntegratedInformationMetric()
        return iit.compute_synergy(state)
