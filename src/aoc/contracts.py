from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class ContractError(ValueError):
    """Raised when a contract is invalid."""


@dataclass(frozen=True)
class InnovationBand:
    min_delta: float
    max_delta: float

    def __post_init__(self) -> None:
        if self.min_delta < 0 or self.max_delta < 0:
            raise ContractError("InnovationBand bounds must be non-negative")
        if self.min_delta > self.max_delta:
            raise ContractError("InnovationBand min_delta must be <= max_delta")


@dataclass(frozen=True)
class SigmaIndex:
    conflict_density: float
    dispersion: float
    revision_elasticity: float
    convergence_slope: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not 0.0 <= value <= 1.0:
                raise ContractError(f"SigmaIndex.{name} must be in [0,1]")

    @property
    def distance_to_transition(self) -> float:
        return (
            self.conflict_density * 0.35
            + self.dispersion * 0.25
            + self.revision_elasticity * 0.25
            + (1 - self.convergence_slope) * 0.15
        )

    @property
    def secondary_diagnostics(self) -> dict[str, float]:
        return {
            "elasticity_raw": self.revision_elasticity,
            "slope_raw": self.convergence_slope,
        }


@dataclass(frozen=True)
class DeltaComponents:
    semantic_delta: float
    structural_delta: float
    functional_delta: float
    weights: tuple[float, float, float] = (0.4, 0.3, 0.3)

    def __post_init__(self) -> None:
        for value in (self.semantic_delta, self.structural_delta, self.functional_delta):
            if value < 0.0:
                raise ContractError("Delta components must be non-negative")
        if abs(sum(self.weights) - 1.0) > 1e-9:
            raise ContractError("Delta component weights must sum to 1")

    @property
    def total(self) -> float:
        w1, w2, w3 = self.weights
        return (
            self.semantic_delta * w1
            + self.structural_delta * w2
            + self.functional_delta * w3
        )


@dataclass(frozen=True)
class AuditResult:
    passed: bool
    confidence: float
    critical_failure: bool
    functional_passed: bool
    structural_passed: bool
    spec_passed: bool
    reason: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ContractError("AuditResult.confidence must be in [0,1]")


@dataclass(frozen=True)
class TaskContract:
    task_prompt: str
    acceptance_criteria: list[str]
    normalized_constraints: dict[str, Any]
    innovation_band: InnovationBand
    evaluator_config: dict[str, Any]
    invariants: list[str]
    max_iterations: int = 10
    coherence_threshold: float = 0.35

    def __post_init__(self) -> None:
        if not self.task_prompt.strip():
            raise ContractError("TaskContract.task_prompt is required")
        if not self.acceptance_criteria:
            raise ContractError("TaskContract.acceptance_criteria must not be empty")
        if self.max_iterations <= 0:
            raise ContractError("TaskContract.max_iterations must be positive")
        if not 0.0 <= self.coherence_threshold <= 1.0:
            raise ContractError("TaskContract.coherence_threshold must be in [0,1]")


@dataclass
class AuditorReliabilityRecord:
    iteration: int
    audit_passed: bool
    audit_confidence: float
    latency_ms: float
    ground_truth: bool | None = None


@dataclass
class AuditorReliabilityTrace:
    records: list[AuditorReliabilityRecord] = field(default_factory=list)

    def append(self, record: AuditorReliabilityRecord) -> None:
        self.records.append(record)

    def precision(self) -> float:
        positives = [r for r in self.records if r.audit_passed]
        if not positives:
            return 0.0
        true_pos = [r for r in positives if r.ground_truth is not False]
        return len(true_pos) / len(positives)

    def avg_latency(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.latency_ms for r in self.records) / len(self.records)

    def false_conservation_rate(self) -> float:
        judged = [r for r in self.records if r.ground_truth is not None]
        if not judged:
            return 0.0
        false_ok = [r for r in judged if r.audit_passed and r.ground_truth is False]
        return len(false_ok) / len(judged)

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [asdict(r) for r in self.records],
            "precision": self.precision(),
            "avg_latency": self.avg_latency(),
            "false_conservation_rate": self.false_conservation_rate(),
        }


def task_contract_from_dict(payload: dict[str, Any]) -> TaskContract:
    band = InnovationBand(
        min_delta=float(payload["innovation_band"]["min_delta"]),
        max_delta=float(payload["innovation_band"]["max_delta"]),
    )
    return TaskContract(
        task_prompt=str(payload["task_prompt"]),
        acceptance_criteria=[str(x) for x in payload["acceptance_criteria"]],
        normalized_constraints=dict(payload.get("normalized_constraints", {})),
        innovation_band=band,
        evaluator_config=dict(payload.get("evaluator_config", {})),
        invariants=[str(x) for x in payload.get("invariants", ["artifact_is_json"])],
        max_iterations=int(payload.get("max_iterations", 10)),
        coherence_threshold=float(payload.get("coherence_threshold", 0.35)),
    )
