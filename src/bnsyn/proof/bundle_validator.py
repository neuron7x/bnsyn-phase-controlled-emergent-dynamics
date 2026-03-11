from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema  # type: ignore[import-untyped]

from bnsyn.proof.contracts import EXPORT_PROOF_ARTIFACTS
from bnsyn.proof.evaluate import sha256_file

ROOT = Path(__file__).resolve().parents[3]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def validate_canonical_bundle(artifact_dir: str | Path) -> dict[str, Any]:
    root = Path(artifact_dir)
    errors: list[str] = []
    manifest = _load_json(root / "run_manifest.json")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("manifest artifacts must be object")

    for artifact in EXPORT_PROOF_ARTIFACTS:
        path = root / artifact
        if not path.is_file():
            errors.append(f"missing artifact: {artifact}")
            continue
        if artifact not in artifacts:
            errors.append(f"manifest missing hash entry: {artifact}")
            continue
        if artifact != "run_manifest.json":
            expected = artifacts[artifact]
            if not isinstance(expected, str) or len(expected) != 64 or sha256_file(path) != expected:
                errors.append(f"manifest hash mismatch: {artifact}")

    schema_map = {
        "run_manifest.json": "run-manifest.schema.json",
        "proof_report.json": "proof-report.schema.json",
        "avalanche_report.json": "avalanche-report.schema.json",
        "avalanche_fit_report.json": "avalanche-fit-report.schema.json",
        "robustness_report.json": "robustness-report.schema.json",
        "envelope_report.json": "envelope-report.schema.json",
        "phase_space_report.json": "phase-space-report.schema.json",
    }
    for artifact, schema_name in schema_map.items():
        path = root / artifact
        if not path.exists():
            continue
        schema = _load_json(ROOT / "schemas" / schema_name)
        try:
            jsonschema.validate(instance=_load_json(path), schema=schema)
        except jsonschema.ValidationError as exc:
            errors.append(f"{artifact} schema violation at {exc.json_path}: {exc.message}")

    return {"status": "PASS" if not errors else "FAIL", "errors": errors}
