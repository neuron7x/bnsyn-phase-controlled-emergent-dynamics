from __future__ import annotations

from dataclasses import dataclass

from .contracts import AuditResult, InnovationBand, SigmaIndex


@dataclass(frozen=True)
class TerminationDecision:
    status: str
    stop_reason: str


class TerminationOracle:
    def evaluate(
        self,
        *,
        iteration: int,
        max_iterations: int,
        delta: float,
        band: InnovationBand,
        sigma: SigmaIndex,
        audit: AuditResult,
        coherence_threshold: float,
        invariants_ok: bool,
    ) -> TerminationDecision:
        if not invariants_ok:
            return TerminationDecision("FAIL", "critical_failure")
        if audit.critical_failure:
            return TerminationDecision("FAIL", "critical_failure")
        if iteration >= max_iterations:
            return TerminationDecision("MAX_ITER", "max_iterations")
        if delta > band.max_delta:
            return TerminationDecision("INCONCLUSIVE", "drift_exceeded")
        if delta < band.min_delta:
            return TerminationDecision("INCONCLUSIVE", "insufficient_progress")
        if not audit.passed:
            return TerminationDecision("INCONCLUSIVE", "inconclusive_audit")
        if sigma.conflict_density >= coherence_threshold:
            return TerminationDecision("INCONCLUSIVE", "insufficient_progress")
        return TerminationDecision("PASS", "productive_emergence")
