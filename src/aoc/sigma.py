from __future__ import annotations

from .contracts import SigmaIndex


class SigmaEngine:
    def compute(self, current_delta: float, previous_delta: float, iteration: int) -> SigmaIndex:
        conflict_density = min(1.0, current_delta)
        dispersion = min(1.0, abs(current_delta - previous_delta))
        revision_elasticity = min(1.0, abs(previous_delta - current_delta) * (1 + iteration * 0.02))
        convergence_slope = 1.0 if previous_delta == 0 else max(0.0, min(1.0, 1 - current_delta / (previous_delta + 1e-9)))
        return SigmaIndex(
            conflict_density=conflict_density,
            dispersion=dispersion,
            revision_elasticity=revision_elasticity,
            convergence_slope=convergence_slope,
        )
