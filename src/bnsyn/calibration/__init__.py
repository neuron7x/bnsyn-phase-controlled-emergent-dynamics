"""Calibration API surface for f-I curve fitting.

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
SPEC.md §P2-10

Claims
------
CLM-0024
"""

from .fit import fit_fI_curve as fit_fI_curve, fit_line as fit_line

__all__ = ["fit_line", "fit_fI_curve"]
