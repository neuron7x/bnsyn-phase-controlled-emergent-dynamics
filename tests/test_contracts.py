from aoc.contracts import InnovationBand, SigmaIndex, TaskContract


def test_innovation_band_validation() -> None:
    band = InnovationBand(min_delta=0.1, max_delta=0.2)
    assert band.min_delta == 0.1


def test_contract_serialization_fields() -> None:
    contract = TaskContract(
        task_prompt="x",
        acceptance_criteria=["a"],
        normalized_constraints={"target_score": 0.5},
        innovation_band=InnovationBand(0.0, 0.5),
        evaluator_config={"deterministic": True},
        invariants=["artifact_is_json"],
    )
    assert contract.max_iterations == 10


def test_sigma_secondary_diagnostics_isolation() -> None:
    sigma = SigmaIndex(0.1, 0.2, 0.3, 0.4)
    assert sigma.secondary_diagnostics == {"elasticity_raw": 0.3, "slope_raw": 0.4}
