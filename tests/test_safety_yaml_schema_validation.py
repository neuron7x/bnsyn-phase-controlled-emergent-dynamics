"""Tests for safety YAML schema validation."""

from __future__ import annotations

from pathlib import Path

from tools.validate_yaml_schema import validate_yaml_schema


ROOT = Path(__file__).resolve().parents[1]


def test_safety_yaml_schema_validation() -> None:
    hazard_errors = validate_yaml_schema(
        ROOT / "docs/safety/hazard_log.yml",
        ROOT / "docs/safety/schemas/hazard_log.schema.json",
    )
    trace_errors = validate_yaml_schema(
        ROOT / "docs/safety/traceability.yml",
        ROOT / "docs/safety/schemas/traceability.schema.json",
    )

    assert hazard_errors == []
    assert trace_errors == []


def test_safety_yaml_schema_rejects_missing_fields(tmp_path: Path) -> None:
    invalid_yaml = tmp_path / "invalid.yml"
    invalid_yaml.write_text("version: '1.0'\n", encoding="utf-8")

    schema_path = ROOT / "docs/safety/schemas/hazard_log.schema.json"
    errors = validate_yaml_schema(invalid_yaml, schema_path)

    assert errors
