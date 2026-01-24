"""Energy regularization API surface.

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
SPEC.md §P1-7

Claims
------
CLM-0021
"""

from .regularization import energy_cost as energy_cost, total_reward as total_reward

__all__ = ["energy_cost", "total_reward"]
