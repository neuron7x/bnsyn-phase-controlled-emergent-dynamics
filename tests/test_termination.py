from aoc.contracts import AuditResult, InnovationBand, SigmaIndex
from aoc.termination import TerminationOracle


def _audit(passed: bool = True, critical: bool = False) -> AuditResult:
    return AuditResult(passed, 1.0 if passed else 0.2, critical, passed, True, True, "x")


def test_no_pass_when_delta_exceeds_band() -> None:
    decision = TerminationOracle().evaluate(
        iteration=1,
        max_iterations=5,
        delta=0.9,
        band=InnovationBand(0.1, 0.5),
        sigma=SigmaIndex(0.1, 0.1, 0.1, 0.9),
        audit=_audit(True),
        coherence_threshold=0.5,
        invariants_ok=True,
    )
    assert decision.status != "PASS"


def test_critical_failure_halts() -> None:
    decision = TerminationOracle().evaluate(
        iteration=1,
        max_iterations=5,
        delta=0.2,
        band=InnovationBand(0.1, 0.5),
        sigma=SigmaIndex(0.1, 0.1, 0.1, 0.9),
        audit=_audit(False, critical=True),
        coherence_threshold=0.5,
        invariants_ok=True,
    )
    assert decision.status == "FAIL"
