from aoc.contracts import AuditResult, DeltaWeights, InnovationBand, SigmaIndex, TaskContract


def test_innovation_band_validation() -> None:
    band = InnovationBand(min_delta=0.1, max_delta=0.2)
    assert band.min_delta == 0.1


def test_contract_defaults_and_shapes() -> None:
    contract = TaskContract(
        task_prompt="x",
        acceptance_criteria=["a"],
        normalized_constraints={"target_score": 0.5},
        innovation_band=InnovationBand(0.0, 0.5),
        evaluator_config={"deterministic": True},
        invariants=["artifact_is_json"],
        artifact_expectations=["status", "score"],
        delta_weights=DeltaWeights(semantic=0.4, structural=0.3, functional=0.3),
    )
    assert contract.max_iterations == 10


def test_sigma_secondary_diagnostics_isolation() -> None:
    sigma = SigmaIndex(0.1, 0.2, 0.3, 0.4)
    assert sigma.secondary_diagnostics == {"elasticity_raw": 0.3, "slope_raw": 0.4}


def test_audit_result_shape() -> None:
    result = AuditResult(
        passed=True,
        confidence=1.0,
        critical_failure=False,
        reasons=["all_gates_passed"],
        checks={"functional": {"passed": True}},
    )
    assert result.reasons[0] == "all_gates_passed"
