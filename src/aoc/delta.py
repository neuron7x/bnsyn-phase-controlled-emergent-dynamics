from __future__ import annotations

from typing import Any

from .contracts import DeltaComponents, TaskContract


class DeltaEngine:
    def compute(self, contract: TaskContract, artifact: dict[str, Any]) -> DeltaComponents:
        target = float(contract.normalized_constraints["target_score"])
        score = float(artifact["score"])
        semantic_delta = abs(score - target)

        expected_keys = sorted(contract.normalized_constraints.get("required_artifact_keys", []))
        present_keys = sorted(artifact.keys())
        missing = len([k for k in expected_keys if k not in present_keys])
        structural_delta = missing / max(1, len(expected_keys))

        functional_threshold = float(contract.normalized_constraints.get("functional_threshold", target))
        functional_delta = max(0.0, functional_threshold - score)
        return DeltaComponents(
            semantic_delta=semantic_delta,
            structural_delta=structural_delta,
            functional_delta=functional_delta,
        )
