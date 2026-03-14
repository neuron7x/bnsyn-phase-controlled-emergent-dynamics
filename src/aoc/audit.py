from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from .contracts import AuditResult, TaskContract


class CrossModelAuditor(Protocol):
    def critique(self, artifact: dict[str, object], contract: TaskContract) -> str:
        ...


@dataclass(frozen=True)
class DeterministicCrossModelAuditor:
    def critique(self, artifact: dict[str, object], contract: TaskContract) -> str:
        _ = contract
        return "stable_local_stub" if "score" in artifact else "missing_score"


@dataclass(frozen=True)
class FunctionalGate:
    def evaluate(self, artifact: dict[str, object], contract: TaskContract) -> tuple[bool, str]:
        threshold = float(contract.normalized_constraints.get("functional_threshold", 0.0))
        score = float(artifact.get("score", -1.0))
        return score >= threshold, f"score={score},threshold={threshold}"


@dataclass(frozen=True)
class StructuralGate:
    def evaluate(self, artifact: dict[str, object], contract: TaskContract) -> tuple[bool, str]:
        required = contract.normalized_constraints.get("required_artifact_keys", [])
        missing = [key for key in required if key not in artifact]
        return len(missing) == 0, f"missing_keys={missing}"


@dataclass(frozen=True)
class SpecComplianceGate:
    def evaluate(self, artifact: dict[str, object], contract: TaskContract) -> tuple[bool, str]:
        status = artifact.get("status")
        valid = status == "candidate"
        return valid, f"status={status}"


class AuditEngine:
    def __init__(self, external: CrossModelAuditor | None = None) -> None:
        self.functional_gate = FunctionalGate()
        self.structural_gate = StructuralGate()
        self.spec_gate = SpecComplianceGate()
        self.external = external or DeterministicCrossModelAuditor()

    def run(self, artifact: dict[str, object], contract: TaskContract) -> tuple[AuditResult, float]:
        started = time.perf_counter()
        functional_ok, functional_detail = self.functional_gate.evaluate(artifact, contract)
        structural_ok, structural_detail = self.structural_gate.evaluate(artifact, contract)
        spec_ok, spec_detail = self.spec_gate.evaluate(artifact, contract)
        critique = self.external.critique(artifact, contract)

        checks: dict[str, Any] = {
            "functional": {"passed": functional_ok, "detail": functional_detail},
            "structural": {"passed": structural_ok, "detail": structural_detail},
            "spec": {"passed": spec_ok, "detail": spec_detail},
            "cross_model_stub": critique,
        }

        reasons: list[str] = []
        if not functional_ok:
            reasons.append("functional_gate_failed")
        if not structural_ok:
            reasons.append("structural_gate_failed")
        if not spec_ok:
            reasons.append("spec_gate_failed")
        if not reasons:
            reasons.append("all_gates_passed")

        critical_failure = not structural_ok
        passed = functional_ok and structural_ok and spec_ok
        confidence = 1.0 if passed else 0.5 if (structural_ok and spec_ok) else 0.2
        latency_ms = (time.perf_counter() - started) * 1000

        return (
            AuditResult(
                passed=passed,
                confidence=confidence,
                critical_failure=critical_failure,
                reasons=reasons,
                checks=checks,
            ),
            latency_ms,
        )
