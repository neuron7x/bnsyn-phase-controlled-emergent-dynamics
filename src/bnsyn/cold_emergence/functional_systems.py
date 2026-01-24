"""Anokhin-style functional systems without affective modulation.

Implements functional system organization based on information processing
rather than motivational drive:
- Afferent synthesis (sensory integration)
- Acceptor of result (prediction vs outcome comparison)
- Cold program execution (action without affective modulation)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ColdFunctionalSystem:
    """Anokhin functional system without motivational/affective modulation.

    Organized INFORMATIONALLY rather than through reward signals.

    Attributes:
        goal_representation: Internal model of desired result
        afferent_synthesis_state: Current integrated sensory state
        error_threshold: Tolerance for accepting results
    """

    goal_representation: np.ndarray
    error_threshold: float = 0.1
    afferent_synthesis_state: np.ndarray | None = None

    def afferent_synthesis(
        self, sensory_input: np.ndarray, memory_context: np.ndarray
    ) -> np.ndarray:
        """Cold afferent synthesis: integrate sensory data with memory.

        Without motivational component - purely informational integration.

        Args:
            sensory_input: Current sensory data
            memory_context: Memory/context information

        Returns:
            Integrated state representation
        """
        # Concatenate inputs
        integrated = np.concatenate([sensory_input, memory_context])

        # Deterministic processing (cold phase, no noise)
        # Simple weighted integration as baseline
        self.afferent_synthesis_state = integrated
        return integrated

    def acceptor_of_result(self, predicted_state: np.ndarray) -> float:
        """Compare predicted result with goal (acceptor of action result).

        Generates CORRECTION signal (not dopamine/reward).

        Args:
            predicted_state: Predicted outcome state

        Returns:
            Error magnitude (correction signal strength)
        """
        error = float(np.linalg.norm(predicted_state - self.goal_representation))
        return error

    def execute_cold_program(
        self,
        action_command: np.ndarray,
        sensory_state: np.ndarray,
        memory_state: np.ndarray,
    ) -> tuple[bool, float]:
        """Execute action without motivational modulation.

        Result evaluated on informational accuracy, not reward.

        Args:
            action_command: Action to execute
            sensory_state: Current sensory input
            memory_state: Current memory context

        Returns:
            Tuple of (success, correction_signal)
        """
        # Afferent synthesis
        prediction = self.afferent_synthesis(sensory_state, memory_state)

        # Acceptor of result comparison
        correction_signal = self.acceptor_of_result(prediction)

        # Success based on informational error, not affective value
        success = correction_signal < self.error_threshold

        return success, correction_signal

    def update_goal(self, new_goal: np.ndarray) -> None:
        """Update goal representation (internal model).

        Args:
            new_goal: New target state
        """
        if new_goal.shape != self.goal_representation.shape:
            raise ValueError(
                f"Goal shape mismatch: expected {self.goal_representation.shape}, "
                f"got {new_goal.shape}"
            )
        self.goal_representation = new_goal
