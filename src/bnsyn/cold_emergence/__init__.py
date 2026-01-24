"""Cold emergence components for BN-Syn system.

Cold emergence refers to deterministic, information-driven system organization
without affective/motivational modulation. This module implements:

- Attractor stabilization through phase control
- Information integration metrics (IIT-inspired Phi)
- Anokhin-style functional systems without motivational drive
- Formal validation of cold emergent properties
"""

from __future__ import annotations

from bnsyn.cold_emergence.controller import ColdPhaseAttractorController
from bnsyn.cold_emergence.functional_systems import ColdFunctionalSystem
from bnsyn.cold_emergence.information_metrics import IntegratedInformationMetric
from bnsyn.cold_emergence.validator import ColdEmergenceValidator

__all__ = [
    "ColdPhaseAttractorController",
    "ColdFunctionalSystem",
    "IntegratedInformationMetric",
    "ColdEmergenceValidator",
]
