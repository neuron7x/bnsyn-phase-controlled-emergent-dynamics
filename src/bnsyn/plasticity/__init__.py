"""Plasticity API surface for STDP and three-factor learning.

Parameters
----------
None

Returns
-------
None

Determinism
-----------
Deterministic under fixed inputs.

SPEC
----
SPEC.md §P0-3

Claims
------
CLM-0004, CLM-0005
"""

from .stdp import stdp_kernel as stdp_kernel
from .three_factor import (
    EligibilityTraces as EligibilityTraces,
    NeuromodulatorTrace as NeuromodulatorTrace,
    three_factor_update as three_factor_update,
)

__all__ = ["stdp_kernel", "EligibilityTraces", "NeuromodulatorTrace", "three_factor_update"]
