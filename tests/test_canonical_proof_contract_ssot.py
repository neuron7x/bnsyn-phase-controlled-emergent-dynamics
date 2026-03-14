from __future__ import annotations

import json
from pathlib import Path

from bnsyn.paths import runtime_file
from bnsyn.proof.bundle_validator import validate_canonical_bundle
from bnsyn.proof.contracts import (
    BASE_ARTIFACTS,
    CANONICAL_BASE_COMMAND,
    CANONICAL_BASE_CONTRACT,
    CANONICAL_EXPORT_PROOF_COMMAND,
    CANONICAL_EXPORT_PROOF_CONTRACT,
    CONTRACT,
    EXPORT_PROOF_ARTIFACTS,
    contract_artifact_specs,
    contract_required_schemas,
)
from scripts.validate_canonical_proof_contract import validate


def test_contract_modes_match_runtime_constants() -> None:
    required = CONTRACT["required_artifacts"]
    assert tuple(required[CANONICAL_BASE_CONTRACT]) == BASE_ARTIFACTS
    assert tuple(required[CANONICAL_EXPORT_PROOF_CONTRACT]) == EXPORT_PROOF_ARTIFACTS


def test_schema_refs_resolve() -> None:
    artifacts = contract_artifact_specs()
    schemas = contract_required_schemas()
    for filename, spec in artifacts.items():
        schema_ref = spec["schema"]
        if schema_ref is None:
            continue
        assert schema_ref in schemas
        schema_path = runtime_file(schemas[schema_ref]["path"])
        assert schema_path.exists(), f"missing schema for {filename}"


def test_validation_gates_g4_matches_ssot() -> None:
    gates = json.loads(runtime_file("ci/validation_gates.json").read_text(encoding="utf-8"))
    by_id = {entry["id"]: entry for entry in gates["registry"]}
    required = by_id["G4_core_artifacts_complete"]["threshold"]["required_artifacts_by_mode"]
    assert required == CONTRACT["required_artifacts"]


def test_run_manifest_schema_modes_match_ssot() -> None:
    schema = json.loads(runtime_file("schemas/run-manifest.schema.json").read_text(encoding="utf-8"))
    assert schema["properties"]["cmd"]["enum"] == [CANONICAL_BASE_COMMAND, CANONICAL_EXPORT_PROOF_COMMAND]
    assert schema["properties"]["bundle_contract"]["enum"] == [CANONICAL_BASE_CONTRACT, CANONICAL_EXPORT_PROOF_CONTRACT]

    true_rule = schema["allOf"][0]["then"]["properties"]
    false_rule = schema["allOf"][1]["then"]["properties"]
    assert true_rule["cmd"]["const"] == CANONICAL_EXPORT_PROOF_COMMAND
    assert true_rule["bundle_contract"]["const"] == CANONICAL_EXPORT_PROOF_CONTRACT
    assert false_rule["cmd"]["const"] == CANONICAL_BASE_COMMAND
    assert false_rule["bundle_contract"]["const"] == CANONICAL_BASE_CONTRACT


def test_runtime_contract_resource_resolves() -> None:
    contract_path = runtime_file("ci/canonical_proof_contract.json")
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    assert payload["contract_name"] == CONTRACT["contract_name"]


def test_bundle_validator_respects_manifest_mode(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "bundle"
    artifact_dir.mkdir()
    manifest = {
        "schema_version": "1.0.0",
        "cmd": CANONICAL_BASE_COMMAND,
        "bundle_contract": CANONICAL_BASE_CONTRACT,
        "export_proof": False,
        "seed": 1,
        "steps": 1,
        "N": 1,
        "dt_ms": 1.0,
        "duration_ms": 1.0,
        "artifacts": {},
    }
    (artifact_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = validate_canonical_bundle(artifact_dir)
    assert not any(err == "missing artifact: proof_report.json" for err in result["errors"])


def test_validator_script_passes() -> None:
    validate()


def test_docs_mark_json_ssot_authoritative() -> None:
    doc = Path("docs/CANONICAL_PROOF.md").read_text(encoding="utf-8")
    assert "src/bnsyn/resources/ci/canonical_proof_contract.json" in doc


def test_optional_artifacts_match_required_in_modes_empty() -> None:
    payload = CONTRACT
    optional = set(payload["optional_artifacts"])
    derived = {entry["filename"] for entry in payload["artifacts"] if entry["required_in_modes"] == []}
    assert optional == derived


def test_bundle_validator_export_mode_requires_proof_report_entry(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "bundle_export"
    artifact_dir.mkdir()
    manifest = {
        "schema_version": "1.0.0",
        "cmd": CANONICAL_EXPORT_PROOF_COMMAND,
        "bundle_contract": CANONICAL_EXPORT_PROOF_CONTRACT,
        "export_proof": True,
        "seed": 1,
        "steps": 1,
        "N": 1,
        "dt_ms": 1.0,
        "duration_ms": 1.0,
        "artifacts": {},
    }
    (artifact_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = validate_canonical_bundle(artifact_dir)
    assert any("export-proof mode requires proof_report.json manifest entry" in err for err in result["errors"])


def test_bundle_validator_base_mode_rejects_manifest_proof_entry(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "bundle_base_forbidden"
    artifact_dir.mkdir()
    manifest = {
        "schema_version": "1.0.0",
        "cmd": CANONICAL_BASE_COMMAND,
        "bundle_contract": CANONICAL_BASE_CONTRACT,
        "export_proof": False,
        "seed": 1,
        "steps": 1,
        "N": 1,
        "dt_ms": 1.0,
        "duration_ms": 1.0,
        "artifacts": {},
    }
    (artifact_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    manifest["artifacts"]["proof_report.json"] = "0" * 64
    (artifact_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = validate_canonical_bundle(artifact_dir)
    assert any("base mode forbids proof_report.json manifest entry" in err for err in result["errors"])
