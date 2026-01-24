"""Temperature scheduling API surface.

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
SPEC.md §P1-5

Claims
------
CLM-0019
"""

from .schedule import TemperatureSchedule as TemperatureSchedule, gate_sigmoid as gate_sigmoid

__all__ = ["TemperatureSchedule", "gate_sigmoid"]
