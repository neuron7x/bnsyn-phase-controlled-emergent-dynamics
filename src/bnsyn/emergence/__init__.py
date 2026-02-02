"""Emergent dynamics and attractor crystallization subpackage.

Parameters
----------
None

Returns
-------
None

Notes
-----
Implements attractor detection and crystallization tracking.

References
----------
docs/emergence_tracking.md
"""

from .crystallizer import (
    Attractor,
    AttractorCrystallizer,
    CrystallizationState,
    Phase,
)

__all__ = [
    "Attractor",
    "AttractorCrystallizer",
    "CrystallizationState",
    "Phase",
]
