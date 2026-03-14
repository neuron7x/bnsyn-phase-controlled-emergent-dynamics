from __future__ import annotations

from dataclasses import dataclass

from .contracts import InnovationBand


@dataclass
class ConstraintProfile:
    step_size: float


class ConstraintModulator:
    def update(self, profile: ConstraintProfile, delta: float, band: InnovationBand) -> ConstraintProfile:
        if delta > band.max_delta:
            return ConstraintProfile(step_size=max(0.01, profile.step_size * 0.5))
        if delta < band.min_delta:
            return ConstraintProfile(step_size=min(1.0, profile.step_size * 1.25))
        return profile
