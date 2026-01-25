"""Calibration utilities for neuronal transfer functions.

Parameters
----------
None

Returns
-------
None

Notes
-----
Exports linear f-I curve fitting helpers.

References
----------
docs/SPEC.md
"""

from .fit import fit_fI_curve as fit_fI_curve, fit_line as fit_line

__all__ = ["fit_line", "fit_fI_curve"]
