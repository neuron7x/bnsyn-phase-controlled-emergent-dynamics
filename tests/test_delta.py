from aoc.contracts import InnovationBand, TaskContract
from aoc.delta import DeltaEngine


def test_delta_components_exposed() -> None:
    contract = TaskContract(
        task_prompt="x",
        acceptance_criteria=["a"],
        normalized_constraints={
            "target_score": 0.5,
            "functional_threshold": 0.5,
            "required_artifact_keys": ["status", "score"],
        },
        innovation_band=InnovationBand(0.0, 1.0),
        evaluator_config={},
        invariants=["artifact_is_json"],
    )
    artifact = {"status": "candidate", "score": 0.25}
    delta = DeltaEngine().compute(contract, artifact)
    assert delta.semantic_delta == 0.25
    assert delta.structural_delta == 0.0
    assert delta.functional_delta == 0.25
