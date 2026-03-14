from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .contracts import AuditResult, TaskContract


class CrossModelAuditor(Protocol):
    def critique(self, content: str, contract: TaskContract) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class LocalCrossModelAuditorStub:
    def critique(self, content: str, contract: TaskContract) -> dict[str, Any]:
        score = 1.0 if contract.objective.lower() in content.lower() else 0.5
        return {"passed": True, "confidence": score, "note": "deterministic_local_stub"}


def _sections(content: str) -> list[str]:
    return [line[3:].strip() for line in content.splitlines() if line.startswith("## ")]


@dataclass(frozen=True)
class FunctionalGate:
    def evaluate(self, content: str, contract: TaskContract) -> dict[str, Any]:
        constraints = contract.constraints
        checks: dict[str, Any] = {}
        required_sections = constraints["required_sections"]
        observed_sections = _sections(content)

        missing = [s for s in required_sections if s not in observed_sections]
        checks["required_sections_present"] = {
            "required": True,
            "passed": len(missing) == 0,
            "detail": {"missing": missing},
        }

        forbidden_found = [t for t in constraints["forbidden_terms"] if t.lower() in content.lower()]
        checks["forbidden_terms_absent"] = {
            "required": True,
            "passed": len(forbidden_found) == 0,
            "detail": {"found": forbidden_found},
        }

        length = len(content)
        checks["length_within_bounds"] = {
            "required": True,
            "passed": constraints["min_length"] <= length <= constraints["max_length"],
            "detail": {"length": length},
        }
        return checks


@dataclass(frozen=True)
class SpecComplianceGate:
    def evaluate(self, content: str, contract: TaskContract) -> dict[str, Any]:
        must_include = bool(contract.invariants["must_include_objective"])
        objective_included = contract.objective.lower() in content.lower()
        return {
            "objective_included": {
                "required": True,
                "passed": (not must_include) or objective_included,
                "detail": {"must_include": must_include},
            }
        }


@dataclass(frozen=True)
class StructuralGate:
    def evaluate(self, content: str, contract: TaskContract) -> dict[str, Any]:
        lines = content.splitlines()
        return {
            "has_title": {
                "required": True,
                "passed": len(lines) > 0 and lines[0].startswith("# "),
                "detail": {},
            },
            "artifact_type_markdown": {
                "required": True,
                "passed": contract.artifact_type == "markdown_document",
                "detail": {},
            },
        }


class AuditEngine:
    def __init__(self, external: CrossModelAuditor | None = None) -> None:
        self.functional = FunctionalGate()
        self.spec = SpecComplianceGate()
        self.structural = StructuralGate()
        self.external = external or LocalCrossModelAuditorStub()

    def run(self, content: str, contract: TaskContract) -> AuditResult:
        checks: dict[str, Any] = {}
        checks.update(self.functional.evaluate(content, contract))
        checks.update(self.spec.evaluate(content, contract))
        checks.update(self.structural.evaluate(content, contract))
        checks["cross_model_stub"] = self.external.critique(content, contract)

        reasons: list[str] = []
        required_fails = [k for k, v in checks.items() if isinstance(v, dict) and v.get("required") and not v.get("passed")]
        critical_failure = any(k in {"has_title", "artifact_type_markdown", "objective_included"} for k in required_fails)

        if required_fails:
            reasons.append("required_checks_failed")
            reasons.extend(required_fails)
        else:
            reasons.append("all_required_checks_passed")

        passed = len(required_fails) == 0
        confidence = 1.0 if passed else max(0.0, 1 - (len(required_fails) / max(1, len(checks))))

        return AuditResult(
            passed=passed,
            confidence=confidence,
            critical_failure=critical_failure,
            reasons=reasons,
            checks=checks,
        )
