"""Numerical integration API surface.

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
SPEC.md §P2-8

Claims
------
CLM-0022
"""

from .integrators import (
    clamp_exp_arg as clamp_exp_arg,
    euler_step as euler_step,
    exp_decay_step as exp_decay_step,
    rk2_step as rk2_step,
)

__all__ = ["euler_step", "rk2_step", "exp_decay_step", "clamp_exp_arg"]
