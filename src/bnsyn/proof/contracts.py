"""Canonical proof mode contracts and helpers derived from SSOT JSON."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
from typing import Any, Final

from bnsyn.paths import runtime_file

_CONTRACT_PATH: Final = runtime_file("ci/canonical_proof_contract.json")
_EXPECTED_TOP_LEVEL_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "contract_name",
        "contract_version",
        "canonical_command",
        "canonical_entrypoint",
        "artifact_dir",
        "bundle_modes",
        "required_artifacts",
        "optional_artifacts",
        "artifacts",
        "required_schema_set",
        "validation_rules",
        "bundle_invariants",
        "compatibility",
    }
)


def _load_json(path: str) -> dict[str, Any]:
    payload = json.loads(runtime_file(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _load_contract() -> dict[str, Any]:
    payload = json.loads(_CONTRACT_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("canonical_proof_contract.json must contain a JSON object")

    extra = set(payload) - _EXPECTED_TOP_LEVEL_FIELDS
    missing = _EXPECTED_TOP_LEVEL_FIELDS - set(payload)
    if extra:
        raise ValueError(f"canonical proof contract has unknown fields: {sorted(extra)}")
    if missing:
        raise ValueError(f"canonical proof contract missing fields: {sorted(missing)}")

    bundle_modes = payload.get("bundle_modes")
    required_artifacts = payload.get("required_artifacts")
    artifacts = payload.get("artifacts")
    if not isinstance(bundle_modes, dict) or not isinstance(required_artifacts, dict):
        raise ValueError("canonical proof contract bundle_modes/required_artifacts must be objects")
    if not isinstance(artifacts, list):
        raise ValueError("canonical proof contract artifacts must be an array")

    for mode in ("canonical-base", "canonical-export-proof"):
        if mode not in bundle_modes:
            raise ValueError(f"canonical proof contract missing bundle mode: {mode}")
        mode_entry = bundle_modes[mode]
        if not isinstance(mode_entry, dict):
            raise ValueError(f"canonical proof contract mode '{mode}' must be object")
        if not isinstance(mode_entry.get("command"), str) or not isinstance(mode_entry.get("export_proof"), bool):
            raise ValueError(f"canonical proof contract mode '{mode}' missing command/export_proof")
        mode_artifacts = required_artifacts.get(mode)
        if not isinstance(mode_artifacts, list) or any(not isinstance(item, str) for item in mode_artifacts):
            raise ValueError(f"canonical proof contract required_artifacts '{mode}' must be list[str]")

    canonical_command = payload.get("canonical_command")
    if not isinstance(canonical_command, str):
        raise ValueError("canonical proof contract canonical_command must be string")
    if canonical_command != bundle_modes["canonical-export-proof"]["command"]:
        raise ValueError("canonical proof contract canonical_command must equal canonical-export-proof command")
    if bundle_modes["canonical-base"].get("export_proof") is not False:
        raise ValueError("canonical-base mode export_proof must be false")
    if bundle_modes["canonical-export-proof"].get("export_proof") is not True:
        raise ValueError("canonical-export-proof mode export_proof must be true")

    filenames: set[str] = set()
    for entry in artifacts:
        if not isinstance(entry, dict):
            raise ValueError("canonical proof contract artifacts entry must be object")
        required_keys = {"filename", "category", "description", "required_in_modes", "schema"}
        if set(entry) != required_keys:
            raise ValueError("canonical proof contract artifact entries must use exact keys")
        filename = entry.get("filename")
        if not isinstance(filename, str):
            raise ValueError("canonical proof contract artifact filename must be string")
        if filename in filenames:
            raise ValueError(f"canonical proof contract duplicate artifact filename: {filename}")
        required_in_modes = entry.get("required_in_modes")
        if not isinstance(required_in_modes, list) or any(not isinstance(m, str) for m in required_in_modes):
            raise ValueError("canonical proof contract required_in_modes must be list[str]")
        for mode_id in required_in_modes:
            if mode_id not in {"canonical-base", "canonical-export-proof"}:
                raise ValueError(f"canonical proof contract artifact {filename} has unknown mode in required_in_modes: {mode_id}")
        filenames.add(filename)

    for mode, mode_artifacts in required_artifacts.items():
        if isinstance(mode_artifacts, list):
            unknown = [item for item in mode_artifacts if item not in filenames]
            if unknown:
                raise ValueError(f"canonical proof contract mode '{mode}' has unknown artifacts: {unknown}")

    for entry in artifacts:
        if not isinstance(entry, dict):
            raise ValueError("canonical proof contract artifact entry must be object")
        filename_raw = entry.get("filename")
        required_in_modes_raw = entry.get("required_in_modes")
        if not isinstance(filename_raw, str) or not isinstance(required_in_modes_raw, list):
            raise ValueError("canonical proof contract artifact entry invalid")
        required_mode_set = set(required_in_modes_raw)
        expected_mode_set = {mode for mode, items in required_artifacts.items() if isinstance(items, list) and filename_raw in items}
        if required_mode_set != expected_mode_set:
            raise ValueError(
                f"canonical proof contract artifact {filename_raw} required_in_modes mismatch vs required_artifacts_by_mode"
            )

    for entry in artifacts:
        if not isinstance(entry, dict):
            raise ValueError("canonical proof contract artifact entry must be object")
        filename_raw = entry.get("filename")
        required_in_modes_raw = entry.get("required_in_modes")
        if not isinstance(filename_raw, str) or not isinstance(required_in_modes_raw, list):
            raise ValueError("canonical proof contract artifact entry invalid")
        required_mode_set = set(required_in_modes_raw)
        expected_mode_set = {
            mode for mode, items in required_artifacts.items() if isinstance(items, list) and filename_raw in items
        }
        if required_mode_set != expected_mode_set:
            raise ValueError(
                f"canonical proof contract artifact {filename_raw} required_in_modes mismatch vs required_artifacts_by_mode"
            )

    optional_artifacts = payload.get("optional_artifacts")
    if not isinstance(optional_artifacts, list) or any(not isinstance(item, str) for item in optional_artifacts):
        raise ValueError("canonical proof contract optional_artifacts must be list[str]")

    optional_set = set(optional_artifacts)
    implied_optional = {
        entry["filename"]
        for entry in artifacts
        if isinstance(entry, dict) and isinstance(entry.get("filename"), str) and entry.get("required_in_modes") == []
    }
    if optional_set != implied_optional:
        raise ValueError("canonical proof contract optional_artifacts mismatch vs artifacts.required_in_modes")

    required_schema_set = payload.get("required_schema_set")
    if not isinstance(required_schema_set, list):
        raise ValueError("canonical proof contract required_schema_set must be an array")
    schema_ids: set[str] = set()
    for schema_entry in required_schema_set:
        if not isinstance(schema_entry, dict):
            raise ValueError("canonical proof contract required_schema_set entries must be objects")
        sid = schema_entry.get("id")
        if not isinstance(sid, str):
            raise ValueError("canonical proof contract schema id must be string")
        if sid in schema_ids:
            raise ValueError(f"canonical proof contract duplicate schema id: {sid}")
        schema_ids.add(sid)

    used_schema_ids = {
        entry["schema"]
        for entry in artifacts
        if isinstance(entry, dict) and isinstance(entry.get("schema"), str)
    }
    if not used_schema_ids.issubset(schema_ids):
        raise ValueError("canonical proof contract artifact schema ref missing from required_schema_set")
    if schema_ids - used_schema_ids:
        raise ValueError("canonical proof contract has unused schema definitions")

    return payload


CONTRACT: Final[dict[str, Any]] = _load_contract()
CANONICAL_BASE_CONTRACT: Final[str] = "canonical-base"
CANONICAL_EXPORT_PROOF_CONTRACT: Final[str] = "canonical-export-proof"
CANONICAL_BASE_COMMAND: Final[str] = CONTRACT["bundle_modes"][CANONICAL_BASE_CONTRACT]["command"]
CANONICAL_EXPORT_PROOF_COMMAND: Final[str] = CONTRACT["bundle_modes"][CANONICAL_EXPORT_PROOF_CONTRACT]["command"]
BASE_ARTIFACTS: Final[tuple[str, ...]] = tuple(CONTRACT["required_artifacts"][CANONICAL_BASE_CONTRACT])
EXPORT_PROOF_ARTIFACTS: Final[tuple[str, ...]] = tuple(CONTRACT["required_artifacts"][CANONICAL_EXPORT_PROOF_CONTRACT])


@dataclass(frozen=True)
class ManifestMode:
    bundle_contract: str
    export_proof: bool
    cmd: str


def command_for_export_proof(export_proof: bool) -> str:
    return CANONICAL_EXPORT_PROOF_COMMAND if export_proof else CANONICAL_BASE_COMMAND


def bundle_contract_for_export_proof(export_proof: bool) -> str:
    return CANONICAL_EXPORT_PROOF_CONTRACT if export_proof else CANONICAL_BASE_CONTRACT


def artifacts_for_export_proof(export_proof: bool) -> tuple[str, ...]:
    return EXPORT_PROOF_ARTIFACTS if export_proof else BASE_ARTIFACTS


def contract_required_schemas() -> dict[str, dict[str, str]]:
    schemas = CONTRACT.get("required_schema_set")
    if not isinstance(schemas, list):
        raise ValueError("canonical proof contract required_schema_set must be an array")
    out: dict[str, dict[str, str]] = {}
    for entry in schemas:
        if not isinstance(entry, dict):
            raise ValueError("canonical proof contract schema entries must be objects")
        schema_id = entry.get("id")
        path = entry.get("path")
        schema_version = entry.get("schema_version")
        if not isinstance(schema_id, str) or not isinstance(path, str) or not isinstance(schema_version, str):
            raise ValueError("canonical proof contract schema entry requires id/path/schema_version strings")
        out[schema_id] = {"path": path, "schema_version": schema_version}
    return out


def contract_artifact_specs() -> dict[str, dict[str, Any]]:
    artifacts = CONTRACT.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("canonical proof contract artifacts must be an array")
    result: dict[str, dict[str, Any]] = {}
    for entry in artifacts:
        if not isinstance(entry, dict):
            raise ValueError("canonical proof contract artifact entry must be object")
        filename = entry.get("filename")
        if not isinstance(filename, str):
            raise ValueError("canonical proof contract artifact filename must be string")
        result[filename] = entry
    return result


def mode_from_manifest(manifest: dict[str, Any]) -> tuple[ManifestMode | None, list[str]]:
    errors: list[str] = []

    cmd_raw = manifest.get("cmd")
    contract_raw = manifest.get("bundle_contract")
    export_raw = manifest.get("export_proof")
    artifacts_raw = manifest.get("artifacts")

    cmd: str | None = cmd_raw if isinstance(cmd_raw, str) else None
    bundle_contract: str | None = contract_raw if isinstance(contract_raw, str) else None
    export_proof: bool | None = export_raw if isinstance(export_raw, bool) else None
    artifacts: dict[str, Any] | None = artifacts_raw if isinstance(artifacts_raw, dict) else None

    if cmd is None:
        errors.append("manifest cmd must be string")
    if bundle_contract not in {CANONICAL_BASE_CONTRACT, CANONICAL_EXPORT_PROOF_CONTRACT}:
        errors.append("manifest bundle_contract invalid")
    if export_proof is None:
        errors.append("manifest export_proof must be boolean")
    if artifacts is None:
        errors.append("manifest artifacts must be object")

    if errors:
        return None, errors

    assert artifacts is not None
    assert cmd is not None
    assert bundle_contract is not None
    assert export_proof is not None

    has_proof_entry = "proof_report.json" in artifacts

    if export_proof:
        if cmd != CANONICAL_EXPORT_PROOF_COMMAND:
            errors.append("export-proof mode requires export-proof cmd")
        if bundle_contract != CANONICAL_EXPORT_PROOF_CONTRACT:
            errors.append("export-proof mode requires canonical-export-proof bundle_contract")
        if not has_proof_entry:
            errors.append("export-proof mode requires proof_report.json manifest entry")
    else:
        if cmd != CANONICAL_BASE_COMMAND:
            errors.append("base mode requires base cmd")
        if bundle_contract != CANONICAL_BASE_CONTRACT:
            errors.append("base mode requires canonical-base bundle_contract")
        if has_proof_entry:
            errors.append("base mode forbids proof_report.json manifest entry")

    if errors:
        return None, errors

    return ManifestMode(bundle_contract=bundle_contract, export_proof=export_proof, cmd=cmd), []


MANIFEST_SELF_HASH_PLACEHOLDER = "__RUN_MANIFEST_SELF_HASH__"


def manifest_self_hash(manifest: dict[str, Any]) -> str:
    """Compute deterministic self-hash for run_manifest.json.

    The run_manifest.json artifact entry is normalized to a fixed placeholder before
    hashing to break recursive self-reference while keeping deterministic integrity.
    """
    payload = copy.deepcopy(manifest)
    artifacts_raw = payload.get("artifacts")
    artifacts: dict[str, Any] = dict(artifacts_raw) if isinstance(artifacts_raw, dict) else {}
    artifacts["run_manifest.json"] = MANIFEST_SELF_HASH_PLACEHOLDER
    payload["artifacts"] = artifacts
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
