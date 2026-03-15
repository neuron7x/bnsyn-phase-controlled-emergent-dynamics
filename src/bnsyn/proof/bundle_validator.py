from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema  # type: ignore[import-untyped]

from bnsyn.proof.contracts import (
    artifacts_for_export_proof,
    contract_artifact_specs,
    contract_required_schemas,
    manifest_self_hash,
    mode_from_manifest,
)
from bnsyn.proof.evaluate import sha256_file
from bnsyn.paths import runtime_file


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def validate_canonical_bundle(
    artifact_dir: str | Path, *, require_product_surface: bool = False
) -> dict[str, Any]:
    root = Path(artifact_dir)
    errors: list[str] = []
    manifest = _load_json(root / "run_manifest.json")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("manifest artifacts must be object")

    mode, mode_errors = mode_from_manifest(manifest)
    if mode is None:
        return {"status": "FAIL", "errors": [f"manifest mode invalid: {msg}" for msg in mode_errors]}

    required_artifacts = artifacts_for_export_proof(mode.export_proof)
    if not mode.export_proof and "proof_report.json" in artifacts:
        errors.append("base mode forbids proof_report.json manifest entry")
    if mode.export_proof and "proof_report.json" not in artifacts:
        errors.append("export-proof mode requires proof_report.json manifest entry")

    for artifact in required_artifacts:
        path = root / artifact
        if not path.is_file():
            errors.append(f"missing artifact: {artifact}")
            continue
        if artifact not in artifacts:
            errors.append(f"manifest missing hash entry: {artifact}")
            continue
        expected = artifacts[artifact]
        if not isinstance(expected, str) or len(expected) != 64:
            errors.append(f"manifest hash mismatch: {artifact}")
            continue

        if artifact == "run_manifest.json":
            if expected != manifest_self_hash(manifest):
                errors.append("manifest hash mismatch: run_manifest.json")
            continue

        if sha256_file(path) != expected:
            errors.append(f"manifest hash mismatch: {artifact}")

    schema_specs = contract_required_schemas()
    artifact_specs = contract_artifact_specs()
    for artifact, artifact_spec in artifact_specs.items():
        if not isinstance(artifact_spec, dict):
            raise ValueError("canonical proof contract artifact spec must be object")
        schema_ref = artifact_spec.get("schema")
        if schema_ref is None:
            continue
        if not isinstance(schema_ref, str):
            raise ValueError(f"canonical proof contract schema ref for {artifact} must be string or null")
        schema_spec = schema_specs.get(schema_ref)
        if schema_spec is None:
            raise ValueError(f"canonical proof contract schema ref missing from required_schema_set: {schema_ref}")

        path = root / artifact
        if not path.exists():
            continue

        schema = _load_json(runtime_file(schema_spec["path"]))
        try:
            payload = _load_json(path)
            jsonschema.validate(instance=payload, schema=schema)
            schema_version = payload.get("schema_version")
            expected_version = schema_spec["schema_version"]
            if isinstance(schema_version, str) and schema_version != expected_version:
                errors.append(
                    f"{artifact} schema_version mismatch: expected {expected_version}, got {schema_version}"
                )
        except jsonschema.ValidationError as exc:
            errors.append(f"{artifact} schema violation at {exc.json_path}: {exc.message}")

    if require_product_surface:
        product_summary_path = root / "product_summary.json"
        index_path = root / "index.html"
        summary_metrics_path = root / "summary_metrics.json"
        proof_report_path = root / "proof_report.json"

        if not product_summary_path.is_file():
            errors.append("missing artifact: product_summary.json")
        if not index_path.is_file():
            errors.append("missing artifact: index.html")
        if not summary_metrics_path.is_file():
            errors.append("missing artifact: summary_metrics.json")
        if not proof_report_path.is_file():
            errors.append("missing artifact: proof_report.json")

        if product_summary_path.is_file():
            product_summary = _load_json(product_summary_path)
            summary_metrics: dict[str, Any] | None = None
            proof_report: dict[str, Any] | None = None

            if summary_metrics_path.is_file():
                summary_metrics = _load_json(summary_metrics_path)
            if proof_report_path.is_file():
                proof_report = _load_json(proof_report_path)

            if product_summary.get("profile") != "canonical":
                errors.append("product_summary profile must be canonical")
            if product_summary.get("proof_verdict") != "PASS":
                errors.append("product_summary proof_verdict must be PASS")
            if product_summary.get("status") != product_summary.get("proof_verdict"):
                errors.append("product_summary status must match proof_verdict")
            if proof_report is not None and product_summary.get("proof_verdict") != proof_report.get("verdict"):
                errors.append("product_summary proof_verdict mismatch vs proof_report verdict")
            if product_summary.get("primary_visual") != "emergence_plot.png":
                errors.append("product_summary primary_visual must be emergence_plot.png")
            if product_summary.get("artifact_dir") != root.as_posix():
                errors.append("product_summary artifact_dir mismatch")
            if product_summary.get("bundle_contract_version") != manifest.get("bundle_contract"):
                errors.append("product_summary bundle_contract_version mismatch vs run_manifest")

            seed = product_summary.get("seed")
            if seed != manifest.get("seed"):
                errors.append("product_summary seed mismatch vs run_manifest")
            if summary_metrics is not None and seed != summary_metrics.get("seed"):
                errors.append("product_summary seed mismatch vs summary_metrics")

        if index_path.is_file() and index_path.stat().st_size == 0:
            errors.append("index.html is empty")

    return {"status": "PASS" if not errors else "FAIL", "errors": errors}
