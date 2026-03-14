from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .audit import AuditEngine
from .contracts import AuditResult, AuditorReliabilityRecord, AuditorReliabilityTrace, SigmaIndex, TaskContract
from .delta import DeltaEngine
from .evidence import EvidenceWriter, sha256_json
from .modulator import ConstraintModulator, ConstraintProfile
from .sigma import SigmaEngine
from .state import AOCState
from .termination import TerminationDecision, TerminationOracle
from .zeropoint import ZeroPointManager


class AOCController:
    def __init__(self, contract: TaskContract, run_dir: Path) -> None:
        self.contract = contract
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.zeropoint = ZeroPointManager(run_dir)
        self.delta_engine = DeltaEngine()
        self.sigma_engine = SigmaEngine()
        self.audit_engine = AuditEngine()
        self.oracle = TerminationOracle()
        self.modulator = ConstraintModulator()
        self.evidence = EvidenceWriter(run_dir)

    def _generate_candidate(self, profile: ConstraintProfile, iteration: int) -> dict[str, Any]:
        initial = float(self.contract.normalized_constraints.get("initial_score", 0.0))
        target = float(self.contract.normalized_constraints["target_score"])
        score = min(target, initial + iteration * profile.step_size)
        return {
            "status": "candidate",
            "score": round(score, 6),
            "iteration": iteration,
            "task": self.contract.task_prompt,
        }

    def run(self) -> dict[str, Any]:
        zeropoint_payload = self.zeropoint.materialize(self.contract)
        zeropoint_hash = zeropoint_payload["canonical_baseline_hash"]

        sigma_trace: list[dict[str, Any]] = []
        delta_trace: list[dict[str, Any]] = []
        audit_trace: list[dict[str, Any]] = []
        reliability = AuditorReliabilityTrace()

        profile = ConstraintProfile(
            step_size=float(self.contract.normalized_constraints.get("initial_step_size", 0.1))
        )
        previous_delta = 1.0
        final_decision = TerminationDecision("MAX_ITER", "max_iterations")
        final_artifact: dict[str, Any] = {}
        final_state: AOCState | None = None

        for iteration in range(1, self.contract.max_iterations + 1):
            artifact = self._generate_candidate(profile, iteration)
            artifact_hash = sha256_json(artifact)

            deltas = self.delta_engine.compute(self.contract, artifact)
            sigma = self.sigma_engine.compute(deltas.total, previous_delta, iteration)
            audit, latency_ms = self.audit_engine.run(artifact, self.contract)
            reliability.append(
                AuditorReliabilityRecord(
                    iteration=iteration,
                    audit_passed=audit.passed,
                    audit_confidence=audit.confidence,
                    latency_ms=latency_ms,
                )
            )

            invariants_ok = self._invariants_ok(artifact)
            decision = self.oracle.evaluate(
                iteration=iteration,
                max_iterations=self.contract.max_iterations,
                delta=deltas.total,
                band=self.contract.innovation_band,
                sigma=sigma,
                audit=audit,
                coherence_threshold=self.contract.coherence_threshold,
                invariants_ok=invariants_ok,
            )

            delta_trace.append(
                {
                    "iteration": iteration,
                    "semantic_delta": deltas.semantic_delta,
                    "structural_delta": deltas.structural_delta,
                    "functional_delta": deltas.functional_delta,
                    "total_delta": deltas.total,
                }
            )
            sigma_trace.append(
                {
                    "iteration": iteration,
                    **asdict(sigma),
                    "distance_to_transition": sigma.distance_to_transition,
                    "secondary_diagnostics": sigma.secondary_diagnostics,
                }
            )
            audit_trace.append({"iteration": iteration, **asdict(audit)})

            final_state = AOCState(
                iteration=iteration,
                zeropoint_hash=zeropoint_hash,
                current_artifact_hash=artifact_hash,
                delta_from_zeropoint=deltas.total,
                sigma=sigma,
                audit=audit,
                band=self.contract.innovation_band,
                status="RUNNING",
            )
            final_artifact = artifact
            final_decision = decision
            previous_delta = deltas.total

            if decision.status in {"PASS", "FAIL", "MAX_ITER"}:
                break
            profile = self.modulator.update(profile, deltas.total, self.contract.innovation_band)

        if final_state is None:
            raise RuntimeError("AOC controller produced no state")

        termination_verdict = {
            "status": final_decision.status,
            "stop_reason": final_decision.stop_reason,
            "iteration": final_state.iteration,
            "delta": final_state.delta_from_zeropoint,
            "sigma_distance": final_state.sigma.distance_to_transition,
            "audit_passed": final_state.audit.passed,
            "band": asdict(final_state.band),
        }
        self.evidence.write_json("final_artifact.json", final_artifact)
        self.evidence.write_json(
            "run_summary.json",
            {
                "iterations_executed": final_state.iteration,
                "final_status": final_decision.status,
                "zeropoint_hash": zeropoint_hash,
            },
        )
        self.evidence.write_trace("sigma_trace.json", sigma_trace)
        self.evidence.write_trace("delta_trace.json", delta_trace)
        self.evidence.write_trace("audit_trace.json", audit_trace)
        self.evidence.write_json("auditor_reliability_trace.json", reliability.to_dict())
        self.evidence.write_json("termination_verdict.json", termination_verdict)
        self.evidence.emit_bundle()
        return termination_verdict

    def _invariants_ok(self, artifact: dict[str, Any]) -> bool:
        for invariant in self.contract.invariants:
            if invariant == "artifact_is_json":
                if not isinstance(artifact, dict):
                    return False
            elif invariant == "non_negative_score":
                if float(artifact.get("score", -1)) < 0:
                    return False
            else:
                return False
        return True
