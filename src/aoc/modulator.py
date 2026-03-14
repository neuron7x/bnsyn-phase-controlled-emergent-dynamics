from __future__ import annotations

from dataclasses import asdict, dataclass

from .contracts import InnovationBand


@dataclass
class ConstraintProfile:
    step_size: float


class ConstraintModulator:
    def update(
        self, profile: ConstraintProfile, delta: float, band: InnovationBand
    ) -> tuple[ConstraintProfile, dict[str, object]]:
        if delta > band.max_delta:
            new_profile = ConstraintProfile(step_size=max(0.01, profile.step_size * 0.5))
            return new_profile, {
                "action": "tighten",
                "reason": "drift_exceeded",
                "from": asdict(profile),
                "to": asdict(new_profile),
            }
        if delta < band.min_delta:
            new_profile = ConstraintProfile(step_size=min(1.0, profile.step_size * 1.25))
            return new_profile, {
                "action": "loosen",
                "reason": "insufficient_progress",
                "from": asdict(profile),
                "to": asdict(new_profile),
            }
        return profile, {
            "action": "hold",
            "reason": "within_band",
            "from": asdict(profile),
            "to": asdict(profile),
        }
