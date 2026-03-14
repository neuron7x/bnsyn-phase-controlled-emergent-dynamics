from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from .contracts import AuditResult, TaskContract


class CrossModelAuditor(Protocol):
    def critique(self, artifact: dict[str, object], contract: TaskContract) -> str:
        ...


@dataclass(frozen=True)
class FunctionalGate:
    def evaluate(self, artifact: dict[str, object], contract: TaskContract) -> bool:
        threshold = float(contract.normalized_constraints.get("functional_threshold", 0.0))
        score = float(artifact.get("score", -1.0))
        return score >= threshold


@dataclass(frozen=True)
class StructuralGate:
    def evaluate(self, artifact: dict[str, object], contract: TaskContract) -> bool:
        required = contract.normalized_constraints.get("required_artifact_keys", [])
        return all(key in artifact for key in required)


@dataclass(frozen=True)
class SpecComplianceGate:
    def evaluate(self, artifact: dict[str, object], contract: TaskContract) -> bool:
        status = artifact.get("status")
        return status == "candidate"


class AuditEngine:
    def __init__(self) -> None:
        self.functional_gate = FunctionalGate()
        self.structural_gate = StructuralGate()
        self.spec_gate = SpecComplianceGate()

    def run(self, artifact: dict[str, object], contract: TaskContract) -> tuple[AuditResult, float]:
        started = time.perf_counter()
        functional_passed = self.functional_gate.evaluate(artifact, contract)
        structural_passed = self.structural_gate.evaluate(artifact, contract)
        spec_passed = self.spec_gate.evaluate(artifact, contract)

        critical_failure = not structural_passed
        passed = functional_passed and structural_passed and spec_passed

        confidence = 1.0 if passed else 0.5 if (structural_passed and spec_passed) else 0.2
        reason = "all_gates_passed" if passed else "gate_failure"

        latency_ms = (time.perf_counter() - started) * 1000
        return (
            AuditResult(
                passed=passed,
                confidence=confidence,
                critical_failure=critical_failure,
                functional_passed=functional_passed,
                structural_passed=structural_passed,
                spec_passed=spec_passed,
                reason=reason,
            ),
            latency_ms,
        )
