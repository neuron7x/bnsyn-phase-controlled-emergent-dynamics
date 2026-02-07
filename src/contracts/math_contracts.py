from __future__ import annotations

import math
from typing import Iterable


def assert_non_empty_text(value: str) -> None:
    if not value.strip():
        msg = "empty_text"
        raise ValueError(msg)


def assert_numeric_finite_and_bounded(values: Iterable[float], *, bound: float = 1e12) -> None:
    for value in values:
        if not math.isfinite(value):
            msg = "non_finite_detected"
            raise ValueError(msg)
        if abs(value) > bound:
            msg = "range_violation_abs_gt_bound"
            raise ValueError(msg)
