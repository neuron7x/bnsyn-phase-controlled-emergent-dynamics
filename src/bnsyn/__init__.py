"""BN-Syn public package entry.

Parameters
----------
None

Returns
-------
None

Determinism
-----------
Module import is deterministic; runtime determinism is governed by explicit RNG seeding.

SPEC
----
SPEC.md §P2-9

Claims
------
CLM-0023
"""

from __future__ import annotations

from importlib import metadata

try:
    __version__ = metadata.version("bnsyn")
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__", "rng", "config"]
