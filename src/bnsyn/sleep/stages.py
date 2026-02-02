"""Sleep stages and configuration.

Parameters
----------
None

Returns
-------
None

Notes
-----
Defines sleep stages (WAKE, LIGHT_SLEEP, DEEP_SLEEP, REM) and their configurations.

References
----------
docs/sleep_stack.md
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SleepStage(Enum):
    """Sleep stage enumeration.

    Attributes
    ----------
    WAKE : int
        Awake state
    LIGHT_SLEEP : int
        Light sleep (NREM stage 2)
    DEEP_SLEEP : int
        Deep sleep (NREM stage 3)
    REM : int
        Rapid eye movement sleep
    """

    WAKE = 0
    LIGHT_SLEEP = 1
    DEEP_SLEEP = 2
    REM = 3


@dataclass(frozen=True)
class SleepStageConfig:
    """Configuration for a sleep stage.

    Parameters
    ----------
    stage : SleepStage
        The sleep stage identifier.
    duration_steps : int
        Duration in simulation steps.
    temperature_range : tuple[float, float]
        Temperature range (min, max) for this stage.
    plasticity_gate : float
        Plasticity gating value in [0, 1].
    consolidation_active : bool
        Whether consolidation is active in this stage.
    replay_active : bool
        Whether memory replay is active in this stage.
    replay_noise : float
        Noise level for replay (0 = exact, 1 = high noise).

    Notes
    -----
    Temperature flows to gate_sigmoid for plasticity gating.
    """

    stage: SleepStage
    duration_steps: int
    temperature_range: tuple[float, float]
    plasticity_gate: float
    consolidation_active: bool
    replay_active: bool
    replay_noise: float

    def __post_init__(self) -> None:
        if self.duration_steps <= 0:
            raise ValueError("duration_steps must be positive")
        if not 0.0 <= self.plasticity_gate <= 1.0:
            raise ValueError("plasticity_gate must be in [0, 1]")
        if not 0.0 <= self.replay_noise <= 1.0:
            raise ValueError("replay_noise must be in [0, 1]")
