from __future__ import annotations

import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_tests_inventory_schema_contract() -> None:
    payload = json.loads((REPO_ROOT / "tests_inventory.json").read_text(encoding="utf-8"))

    assert isinstance(payload["generated_by"], str)
    assert isinstance(payload["test_count"], int)
    assert payload["test_count"] >= 1
    assert isinstance(payload["coverage_surface"], dict)
    assert isinstance(payload["reusable_workflow_jobs"], list)
    assert isinstance(payload["tests"], list)
    assert payload["test_count"] == len(payload["tests"])


def test_acceptance_map_contains_no_escape_contract() -> None:
    payload = yaml.safe_load((REPO_ROOT / "acceptance_map.yaml").read_text(encoding="utf-8"))
    no_escape = payload["no_escape_acceptance"]

    assert no_escape["merge_safety_budget_minutes"] == 12
    assert no_escape["determinism_runs"] == 3
    assert no_escape["decision_mode"] == "fail_closed"

    required_checks = no_escape["required_checks"]
    assert required_checks == ["ci-pr-atomic", "workflow-integrity", "determinism", "contracts"]

    forbidden = set(no_escape["forbidden"])
    assert "unpinned_actions" in forbidden
    assert "flaky_tests" in forbidden
    assert "unknown_evidence" in forbidden
