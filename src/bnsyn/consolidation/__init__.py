"""Consolidation API surface for dual-weight synapses.

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
SPEC.md §P1-6

Claims
------
CLM-0010, CLM-0020
"""

from .dual_weight import DualWeights as DualWeights

__all__ = ["DualWeights"]
