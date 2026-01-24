"""Cold phase attractor controller for BN-Syn.

Implements deterministic attractor stabilization through:
- Lyapunov exponent computation for stability analysis
- Phase control via temperature/criticality modulation
- Attractor basin navigation without stochastic exploration
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ColdPhaseAttractorController:
    """Cold phase attractor controller using criticality (σ) for phase control.

    Cold phase = deterministic attractor control through minimization of
    criticality (σ near critical point).

    Attributes:
        target_sigma: Target criticality value (near 1.0 for critical phase)
        lyapunov_trajectory: History of Lyapunov exponents
    """

    target_sigma: float = 1.0
    lyapunov_trajectory: list[float] = field(default_factory=list)

    def compute_lyapunov_exponent(
        self, state: np.ndarray, perturbed_state: np.ndarray, dt: float, eps: float = 1e-8
    ) -> float:
        """Compute Lyapunov exponent from state divergence.

        Lyapunov exp < 0 → cold, deterministic (exponential stability)
        Lyapunov exp ≈ 0 → critical phase (maximal information integration)
        Lyapunov exp > 0 → chaotic, exploratory

        Args:
            state: Current system state
            perturbed_state: State with small perturbation
            dt: Time step
            eps: Perturbation magnitude for numerical stability

        Returns:
            Lyapunov exponent estimate
        """
        divergence = float(np.linalg.norm(state - perturbed_state))
        if divergence < eps:
            return 0.0
        lyap = float(np.log(divergence / eps) / dt)
        self.lyapunov_trajectory.append(lyap)
        return lyap

    def stabilize_cold_attractor(
        self,
        current_state: np.ndarray,
        target_state: np.ndarray,
        current_sigma: float,
        force_gain: float = 0.1,
    ) -> tuple[np.ndarray, float]:
        """Guide system to cold attractor without stochastic exploration.

        Applies deterministic force toward target attractor state and
        returns suggested temperature reduction factor.

        Args:
            current_state: Current system state
            target_state: Target attractor state
            current_sigma: Current criticality value
            force_gain: Strength of attractor pulling force

        Returns:
            Tuple of (deterministic_force, temperature_reduction_factor)
        """
        # Calculate deviation from target
        deviation = target_state - current_state

        # Deterministic force proportional to deviation
        force = force_gain * deviation

        # Temperature reduction factor based on distance from target sigma
        sigma_error = abs(current_sigma - self.target_sigma)
        temp_reduction = 0.95 if sigma_error > 0.1 else 0.99

        return force, temp_reduction

    def is_in_cold_phase(self, sigma: float, lyapunov: float) -> bool:
        """Check if system is in cold phase.

        Cold phase criteria:
        - Criticality near target (sigma ≈ target_sigma)
        - Negative or near-zero Lyapunov exponent

        Args:
            sigma: Current criticality value
            lyapunov: Current Lyapunov exponent

        Returns:
            True if system is in cold phase
        """
        sigma_close = abs(sigma - self.target_sigma) < 0.1
        stable = lyapunov < 0.1
        return sigma_close and stable
