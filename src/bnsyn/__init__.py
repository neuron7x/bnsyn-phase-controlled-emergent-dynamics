"""BN-Syn package entry point and version metadata.

Parameters
----------
None

Returns
-------
None

Notes
-----
This module exposes package version metadata and lazily re-exports canonical
API functions from :mod:`bnsyn.api`.

References
----------
docs/SPEC.md
"""

from __future__ import annotations

from importlib import metadata
from typing import Any

try:
    __version__ = metadata.version("bnsyn")
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__", "run", "phase_atlas", "sleep_stack"]


def __getattr__(name: str) -> Any:
    """Lazily expose canonical API callables from ``bnsyn.api``."""
    if name in {"run", "phase_atlas", "sleep_stack"}:
        from bnsyn.api import phase_atlas, run, sleep_stack

        exports = {
            "run": run,
            "phase_atlas": phase_atlas,
            "sleep_stack": sleep_stack,
        }
        return exports[name]
    raise AttributeError(f"module 'bnsyn' has no attribute {name!r}")
