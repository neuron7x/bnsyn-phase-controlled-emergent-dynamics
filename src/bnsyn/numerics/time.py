"""Time/step conversion helpers for deterministic simulation contracts."""

from __future__ import annotations

from math import isclose


def compute_steps_exact(duration_ms: float, dt_ms: float) -> int:
    """Return integer simulation steps and fail closed on precision loss."""
    if dt_ms <= 0:
        raise ValueError(f"dt_ms must be positive, got {dt_ms}")
    steps_float = duration_ms / dt_ms
    steps_int = int(round(steps_float))
    if not isclose(steps_float, steps_int, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            f"Non-integer steps: {duration_ms}ms / {dt_ms}ms = {steps_float}. "
            "Duration must be an exact multiple of dt."
        )
    return steps_int
