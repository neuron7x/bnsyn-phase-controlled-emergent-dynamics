from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_sse_sdo_config_has_required_fields() -> None:
    config_path = ROOT / ".github" / "sse_sdo.yml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert payload["protocol"] == "SSE-SDO-2026.04"
    assert payload["toolchain"]["python"] == "3.11"
    assert payload["policy"]["evidence_required_ratio_P0"] == 1.0
    assert payload["policy"]["contradictions_required"] == 0
    assert payload["tests"]["coverage"]["line_pct_min"] == 0.90
    assert payload["tests"]["coverage"]["branch_pct_min"] == 0.80
    assert payload["perf"]["baseline_required"] is True
    assert payload["flags"]["registry_path"] == "artifacts/sse_sdo/architecture/FLAGS.md"


def test_sse_sdo_artifact_tree_minimum_files_exist() -> None:
    required_paths = [
        "artifacts/sse_sdo/meta/REPO_FINGERPRINT.json",
        "artifacts/sse_sdo/meta/ENV_SNAPSHOT.json",
        "artifacts/sse_sdo/meta/RUN_MANIFEST.json",
        "artifacts/sse_sdo/architecture/SUBSYSTEM_BOUNDARY.md",
        "artifacts/sse_sdo/architecture/CONTRACTS.md",
        "artifacts/sse_sdo/architecture/FLAGS.md",
        "artifacts/sse_sdo/architecture/SSOT_DELTA.md",
        "artifacts/sse_sdo/tests/TEST_PLAN.md",
        "artifacts/sse_sdo/tests/CONTRACT_TEST_MAP.json",
        "artifacts/sse_sdo/perf/PERF_REPORT.md",
        "artifacts/sse_sdo/ci/REQUIRED_CHECKS_MANIFEST.json",
        "artifacts/sse_sdo/ci/DRIFT_REPORT.json",
        "artifacts/sse_sdo/quality.json",
        "artifacts/sse_sdo/EVIDENCE_INDEX.md",
        "artifacts/sse_sdo/CONTRADICTIONS.json",
    ]

    for relpath in required_paths:
        assert (ROOT / relpath).exists(), relpath

    quality = json.loads((ROOT / "artifacts/sse_sdo/quality.json").read_text(encoding="utf-8"))
    assert quality["verdict"] == "PENDING"
    assert quality["contradictions"] == 0
