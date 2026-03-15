"""Validate canonical proof consumers against SSOT contract."""

from __future__ import annotations

import json
import tomllib

import yaml
from pathlib import Path
from typing import Any

from bnsyn.paths import runtime_file
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

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_runtime_derivation() -> None:
    required_by_mode = CONTRACT["required_artifacts"]
    _assert(tuple(required_by_mode[CANONICAL_BASE_CONTRACT]) == BASE_ARTIFACTS, "BASE_ARTIFACTS drift from SSOT")
    _assert(
        tuple(required_by_mode[CANONICAL_EXPORT_PROOF_CONTRACT]) == EXPORT_PROOF_ARTIFACTS,
        "EXPORT_PROOF_ARTIFACTS drift from SSOT",
    )


def _validate_contract_semantics() -> None:
    modes = CONTRACT["bundle_modes"]
    _assert(
        CONTRACT["canonical_command"] == modes[CANONICAL_EXPORT_PROOF_CONTRACT]["command"],
        "canonical_command must equal canonical-export-proof command",
    )
    _assert(modes[CANONICAL_BASE_CONTRACT]["export_proof"] is False, "canonical-base export_proof must be false")
    _assert(
        modes[CANONICAL_EXPORT_PROOF_CONTRACT]["export_proof"] is True,
        "canonical-export-proof export_proof must be true",
    )

    artifacts = CONTRACT["artifacts"]
    artifact_names = [item["filename"] for item in artifacts]
    _assert(len(artifact_names) == len(set(artifact_names)), "duplicate artifact filenames in SSOT")

    required_schemas = contract_required_schemas()
    for item in artifacts:
        schema_ref = item.get("schema")
        if schema_ref is not None:
            _assert(schema_ref in required_schemas, f"schema ref missing from required_schema_set: {schema_ref}")
        modes_list = item.get("required_in_modes")
        _assert(isinstance(modes_list, list), f"required_in_modes must be list for {item['filename']}")

    required_by_mode = CONTRACT["required_artifacts"]
    for mode_name, mode_artifacts in required_by_mode.items():
        for artifact in mode_artifacts:
            _assert(artifact in artifact_names, f"required_artifact not declared in artifacts list: {artifact}")


def _validate_gate_registry() -> None:
    repo_gate_path = REPO_ROOT / "ci" / "validation_gates.json"
    packaged_gate_path = runtime_file("ci/validation_gates.json")
    repo_gate = _load_json(repo_gate_path)
    packaged_gate = _load_json(packaged_gate_path)
    _assert(repo_gate == packaged_gate, "repo and packaged validation_gates.json diverged")

    registry = packaged_gate.get("registry")
    if not isinstance(registry, list):
        raise ValueError("validation_gates registry must be array")
    by_id = {entry.get("id"): entry for entry in registry if isinstance(entry, dict)}
    g4 = by_id.get("G4_core_artifacts_complete")
    if not isinstance(g4, dict):
        raise ValueError("G4_core_artifacts_complete missing")
    threshold = g4.get("threshold")
    if not isinstance(threshold, dict):
        raise ValueError("G4 threshold missing")
    g4_required = threshold.get("required_artifacts_by_mode")
    if not isinstance(g4_required, dict):
        raise ValueError("G4 required_artifacts_by_mode missing")
    _assert(
        g4_required == CONTRACT["required_artifacts"],
        "validation_gates G4 required_artifacts_by_mode drift from canonical_proof_contract.json",
    )


def _validate_schema_refs() -> None:
    artifact_specs = contract_artifact_specs()
    required_schemas = contract_required_schemas()
    for artifact, spec in artifact_specs.items():
        schema_ref = spec.get("schema")
        if schema_ref is None:
            continue
        _assert(schema_ref in required_schemas, f"artifact schema ref missing from required_schema_set: {artifact}")


def _validate_run_manifest_schema_alignment() -> None:
    repo_schema_path = REPO_ROOT / "src" / "bnsyn" / "resources" / "schemas" / "run-manifest.schema.json"
    runtime_schema_path = runtime_file("schemas/run-manifest.schema.json")
    repo_schema = _load_json(repo_schema_path)
    runtime_schema = _load_json(runtime_schema_path)
    _assert(repo_schema == runtime_schema, "repo and runtime run-manifest schema diverged")

    properties = runtime_schema.get("properties")
    if not isinstance(properties, dict):
        raise ValueError("run-manifest schema missing properties")

    cmd_enum = properties.get("cmd", {}).get("enum")
    contract_enum = properties.get("bundle_contract", {}).get("enum")
    expected_cmd_enum = [CANONICAL_BASE_COMMAND, CANONICAL_EXPORT_PROOF_COMMAND]
    expected_contract_enum = [CANONICAL_BASE_CONTRACT, CANONICAL_EXPORT_PROOF_CONTRACT]
    _assert(cmd_enum == expected_cmd_enum, "run-manifest schema cmd.enum drift from SSOT")
    _assert(contract_enum == expected_contract_enum, "run-manifest schema bundle_contract.enum drift from SSOT")

    all_of = runtime_schema.get("allOf")
    if not isinstance(all_of, list) or len(all_of) < 2:
        raise ValueError("run-manifest schema allOf invariants missing")

    true_rule = all_of[0]
    false_rule = all_of[1]
    if not isinstance(true_rule, dict) or not isinstance(false_rule, dict):
        raise ValueError("run-manifest schema allOf entries must be objects")

    true_then = true_rule.get("then", {}).get("properties", {})
    false_then = false_rule.get("then", {}).get("properties", {})
    _assert(
        true_then.get("cmd", {}).get("const") == CANONICAL_EXPORT_PROOF_COMMAND,
        "run-manifest schema export_proof=true cmd const drift from SSOT",
    )
    _assert(
        true_then.get("bundle_contract", {}).get("const") == CANONICAL_EXPORT_PROOF_CONTRACT,
        "run-manifest schema export_proof=true bundle_contract const drift from SSOT",
    )
    _assert(
        false_then.get("cmd", {}).get("const") == CANONICAL_BASE_COMMAND,
        "run-manifest schema export_proof=false cmd const drift from SSOT",
    )
    _assert(
        false_then.get("bundle_contract", {}).get("const") == CANONICAL_BASE_CONTRACT,
        "run-manifest schema export_proof=false bundle_contract const drift from SSOT",
    )


def _validate_packaging_contract_resource() -> None:
    runtime_contract_path = runtime_file("ci/canonical_proof_contract.json")
    _assert(runtime_contract_path.exists(), "runtime canonical_proof_contract.json is not resolvable")

    repo_contract_path = REPO_ROOT / "src" / "bnsyn" / "resources" / "ci" / "canonical_proof_contract.json"
    runtime_payload = _load_json(runtime_contract_path)
    repo_payload = _load_json(repo_contract_path)
    _assert(runtime_payload == repo_payload, "repo and runtime canonical_proof_contract.json diverged")

    pyproject_data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    tool_cfg = pyproject_data.get("tool", {}).get("setuptools", {})
    if not isinstance(tool_cfg, dict):
        raise ValueError("pyproject.toml missing [tool.setuptools] configuration")
    _assert(tool_cfg.get("include-package-data") is True, "setuptools include-package-data must be true")

    package_data = tool_cfg.get("package-data")
    if not isinstance(package_data, dict):
        raise ValueError("pyproject.toml missing [tool.setuptools.package-data]")
    bnsyn_data = package_data.get("bnsyn")
    if not isinstance(bnsyn_data, list):
        raise ValueError("pyproject.toml package-data.bnsyn must be list")
    _assert("resources/**/*.json" in bnsyn_data, "package-data.bnsyn must include resources/**/*.json")


def _workflow_contains_run_command(path: Path, command: str) -> bool:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"workflow yaml must decode to object: {path}")
    jobs = payload.get("jobs")
    if not isinstance(jobs, dict):
        raise ValueError(f"workflow missing jobs: {path}")
    for job in jobs.values():
        if not isinstance(job, dict):
            continue
        steps = job.get("steps")
        if not isinstance(steps, list):
            continue
        for step in steps:
            if isinstance(step, dict) and step.get("run") == command:
                return True
    return False


def _validate_docs_and_workflow_wiring() -> None:
    doc_text = (REPO_ROOT / "docs" / "CANONICAL_PROOF.md").read_text(encoding="utf-8")
    _assert(
        "src/bnsyn/resources/ci/canonical_proof_contract.json" in doc_text,
        "docs/CANONICAL_PROOF.md must point to canonical_proof_contract.json as authoritative source",
    )
    _assert(CANONICAL_EXPORT_PROOF_COMMAND in doc_text, "docs/CANONICAL_PROOF.md missing canonical command")

    spine_path = REPO_ROOT / ".github" / "workflows" / "canonical-proof-spine.yml"
    _assert(
        _workflow_contains_run_command(spine_path, "python -m scripts.validate_canonical_proof_contract"),
        "canonical-proof-spine missing SSOT validator step",
    )
    _assert(
        _workflow_contains_run_command(spine_path, "bnsyn run --profile canonical --plot --export-proof --output artifacts/canonical_run"),
        "canonical-proof-spine missing canonical proof run command",
    )

    ci_atomic_path = REPO_ROOT / ".github" / "workflows" / "ci-pr-atomic.yml"
    _assert(
        _workflow_contains_run_command(ci_atomic_path, "python -m scripts.validate_canonical_proof_contract"),
        "ci-pr-atomic missing SSOT validator step",
    )


def _validate_makefile_wiring() -> None:
    makefile_text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    _assert("validate-proof-contract:" in makefile_text, "Makefile missing validate-proof-contract target")
    lines = makefile_text.splitlines()
    target_idx = next((idx for idx, line in enumerate(lines) if line.startswith("validate-proof-contract:")), None)
    if target_idx is None:
        raise ValueError("Makefile missing validate-proof-contract target")

    recipe_lines: list[str] = []
    for line in lines[target_idx + 1 :]:
        if not line:
            continue
        if line.startswith("\t"):
            recipe_lines.append(line.strip())
            continue
        if not line.startswith(" "):
            break

    _assert(recipe_lines, "Makefile validate-proof-contract target must define recipe commands")
    _assert(
        any("-m scripts.validate_canonical_proof_contract" in line for line in recipe_lines),
        "Makefile validate-proof-contract target must execute python -m scripts.validate_canonical_proof_contract",
    )


def validate() -> None:
    _validate_runtime_derivation()
    _validate_contract_semantics()
    _validate_gate_registry()
    _validate_schema_refs()
    _validate_run_manifest_schema_alignment()
    _validate_packaging_contract_resource()
    _validate_docs_and_workflow_wiring()
    _validate_makefile_wiring()


if __name__ == "__main__":
    validate()
    print("PASS: canonical proof SSOT contract validation")
