"""Synapse API surface for conductance-based dynamics.

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
SPEC.md §P0-2

Claims
------
CLM-0003
"""

from .conductance import (
    ConductanceState as ConductanceState,
    ConductanceSynapses as ConductanceSynapses,
)

__all__ = ["ConductanceState", "ConductanceSynapses"]
