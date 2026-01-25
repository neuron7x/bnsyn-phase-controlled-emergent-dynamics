"""Consolidation subpackage for dual-weight synapses.

Parameters
----------
None

Returns
-------
None

Notes
-----
Exports the dual-weight consolidation model used in SPEC P1-3.

References
----------
docs/SPEC.md#P1-3
"""

from .dual_weight import DualWeights as DualWeights

__all__ = ["DualWeights"]
