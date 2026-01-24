"""Simulation API surface for the reference network.

Parameters
----------
None

Returns
-------
None

Determinism
-----------
Deterministic under fixed seed and parameters.

SPEC
----
SPEC.md §P2-11, §P2-9

Claims
------
CLM-0025, CLM-0023
"""

from .network import (
    Network as Network,
    NetworkParams as NetworkParams,
    run_simulation as run_simulation,
)

__all__ = ["Network", "NetworkParams", "run_simulation"]
