"""Validate experiment results JSON files against the schema."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft7Validator

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "schemas" / "experiment_results.schema.json"
RESULTS_DIR = ROOT / "results" / "temp_ablation_v2"


@pytest.mark.validation
def test_results_schema_temp_ablation_v2() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft7Validator(schema)

    result_files = sorted(
        path
        for path in RESULTS_DIR.glob("*.json")
        if path.name != "manifest.json"
    )
    assert result_files, "Expected result files in results/temp_ablation_v2"

    for path in result_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        errors = sorted(validator.iter_errors(payload), key=lambda err: err.path)
        if errors:
            formatted = "; ".join(
                f"{'.'.join(str(p) for p in err.path) or '<root>'}: {err.message}"
                for err in errors
            )
            raise AssertionError(f"Schema validation failed for {path}: {formatted}")
