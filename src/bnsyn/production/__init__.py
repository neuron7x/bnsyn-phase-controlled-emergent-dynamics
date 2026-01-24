"""Production-oriented utilities for BN-Syn.

Parameters
----------
None

Returns
-------
None

Determinism
-----------
Deterministic under fixed inputs and fixed parameters.

SPEC
----
SPEC.md §P2-11

Claims
------
None
"""

from .adex import AdExNeuron, AdExParams
from .connectivity import ConnectivityConfig, build_connectivity

__all__ = [
    "AdExParams",
    "AdExNeuron",
    "ConnectivityConfig",
    "build_connectivity",
]
