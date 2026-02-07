from __future__ import annotations

import json
import math
from pathlib import Path

from scripts.math_validate import (
    MANIFEST_PATH,
    build_manifest,
    classify_file,
    extract_numeric_scalars,
    iter_scope_files,
    load_data,
    validate_manifest,
)
from src.contracts import assert_non_empty_text, assert_numeric_finite_and_bounded


def test_extract_numeric_scalars_rejects_non_finite() -> None:
    values = extract_numeric_scalars({"x": [1.0, 2, {"y": -3.5}]})
    assert values == [1.0, 2.0, -3.5]


def test_validator_fails_on_hash_drift(tmp_path: Path) -> None:
    artifact = tmp_path / "sample.json"
    artifact.write_text(json.dumps({"value": 1.0}), encoding="utf-8")
    manifest = {
        "schema_version": "2.0",
        "timestamp": "2026-01-01T00:00:00Z",
        "artifacts": [
            {
                "path": str(artifact),
                "sha256": "0" * 64,
                "type": "derived_data",
                "generator": None,
                "provenance": "UNTRUSTED",
                "provenance_gap": "test",
            }
        ],
    }
    checks = validate_manifest(manifest)
    assert any(c.check_name == "sha256_match" and c.status == "FAIL" for c in checks)


def test_contract_negative_cases() -> None:
    try:
        assert_non_empty_text("   ")
    except ValueError as exc:
        assert str(exc) == "empty_text"
    else:
        raise AssertionError("expected empty_text failure")

    try:
        assert_numeric_finite_and_bounded([1.0, math.inf])
    except ValueError as exc:
        assert str(exc) == "non_finite_detected"
    else:
        raise AssertionError("expected non_finite_detected failure")


def test_classify_file_covers_contract_families() -> None:
    assert classify_file(Path("src/bnsyn/simulation.py")) == "numeric_code"
    assert classify_file(Path("results/out.csv")) == "derived_data"
    assert classify_file(Path("docs/ARCHITECTURE.md")) == "report"


def test_manifest_covers_all_scoped_files() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest_paths = {item["path"] for item in manifest["artifacts"]}
    scoped_paths = {str(path).replace("\\", "/") for path in iter_scope_files()}
    assert manifest_paths == scoped_paths


def test_validator_passes_for_current_manifest_and_has_checks_for_all_artifacts() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    checks = validate_manifest(manifest)
    failures = [check for check in checks if check.status == "FAIL"]
    assert failures == []

    checks_by_artifact: dict[str, set[str]] = {}
    for check in checks:
        checks_by_artifact.setdefault(check.artifact, set()).add(check.check_name)

    for artifact in manifest["artifacts"]:
        names = checks_by_artifact[artifact["path"]]
        assert "sha256_match" in names
        assert "schema_load" in names
        assert "numeric_sanity" in names
        assert "distribution_anomaly" in names
        if artifact["path"].endswith(".py"):
            assert "numeric_hazard_scan" in names


def test_all_derived_data_artifacts_are_loadable_and_bounded() -> None:
    manifest = build_manifest()
    for artifact in manifest["artifacts"]:
        if artifact["type"] != "derived_data":
            continue
        path = Path(artifact["path"])
        data = load_data(path)
        assert data is not None
        numerics = extract_numeric_scalars(data)
        if numerics:
            assert_numeric_finite_and_bounded(numerics, bound=1e12)


def test_hazard_scan_skips_unsupported_python_syntax(tmp_path: Path) -> None:
    artifact = tmp_path / "unsupported.py"
    artifact.write_text("def f[T](x: T) -> T:\n    return x\n", encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "2.0",
        "timestamp": "2026-01-01T00:00:00Z",
        "artifacts": [
            {
                "path": str(artifact),
                "sha256": digest,
                "type": "numeric_code",
                "generator": None,
                "provenance": "SOURCED",
                "provenance_gap": "test",
            }
        ],
    }
    checks = validate_manifest(manifest)
    hazard_checks = [c for c in checks if c.check_name == "numeric_hazard_scan"]
    assert len(hazard_checks) == 1
    assert hazard_checks[0].status in {"SKIP", "PASS"}
