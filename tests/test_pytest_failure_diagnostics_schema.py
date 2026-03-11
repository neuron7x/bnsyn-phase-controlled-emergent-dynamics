"""Schema validation tests for pytest failure diagnostics."""

from __future__ import annotations

from pathlib import Path

import pytest

from bnsyn.qa.pytest_failure_diagnostics import validate_payload

SCHEMA = Path("schemas/pytest-failure-diagnostics.schema.json")


def test_valid_payload_passes_schema() -> None:
    payload = {
        "schema_version": "1.0.0",
        "status": "clean",
        "pytest_exit_code": 0,
        "summary": {"tests_total": 1, "failures": 0, "errors": 0, "skipped": 0},
        "failures": [],
    }
    validate_payload(payload, SCHEMA)


def test_malformed_payload_fails_schema() -> None:
    payload = {
        "schema_version": "1.0.0",
        "status": "clean",
        "pytest_exit_code": "0",
        "summary": {"tests_total": 1, "failures": 0, "errors": 0, "skipped": 0},
        "failures": [],
    }
    with pytest.raises(Exception):
        validate_payload(payload, SCHEMA)


def test_input_error_payload_must_include_input_error_block() -> None:
    payload = {
        "schema_version": "1.0.0",
        "status": "input_error",
        "pytest_exit_code": 2,
        "summary": {"tests_total": 0, "failures": 0, "errors": 0, "skipped": 0},
        "failures": [],
    }
    with pytest.raises(Exception):
        validate_payload(payload, SCHEMA)

    payload["input_error"] = {"type": "ParseError", "message": "bad xml"}
    validate_payload(payload, SCHEMA)
