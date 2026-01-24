"""Neuron API surface for AdEx dynamics.

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
SPEC.md §P0-1, §P2-8

Claims
------
CLM-0001, CLM-0002
"""

from .adex import (
    AdExState as AdExState,
    IntegrationMetrics as IntegrationMetrics,
    adex_step as adex_step,
    adex_step_with_error_tracking as adex_step_with_error_tracking,
)

__all__ = ["AdExState", "IntegrationMetrics", "adex_step", "adex_step_with_error_tracking"]
