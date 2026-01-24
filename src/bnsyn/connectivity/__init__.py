"""Connectivity management (sparse and dense).

Parameters
----------
None

Returns
-------
None

Determinism
-----------
Deterministic under fixed inputs and RNG state.

SPEC
----
SPEC.md §P2-11

Claims
------
None
"""

from __future__ import annotations

from bnsyn.connectivity.sparse import SparseConnectivity, build_random_connectivity

__all__ = ["SparseConnectivity", "build_random_connectivity"]
