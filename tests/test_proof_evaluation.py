from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import bnsyn.proof.evaluate as proof_evaluate
from bnsyn.experiments.declarative import run_canonical_live_bundle
from bnsyn.proof.evaluate import evaluate_all_gates

ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cli_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_path if not existing else f"{src_path}:{existing}"
    return env


def _assert_consistent_bundle(out_dir: Path) -> None:
    report = _load_json(out_dir / "proof_report.json")
    manifest = _load_json(out_dir / "run_manifest.json")
    manifest_hash = manifest["artifacts"]["proof_report.json"]
    actual_hash = hashlib.sha256((out_dir / "proof_report.json").read_bytes()).hexdigest()
    assert manifest_hash == actual_hash
    reevaluated = evaluate_all_gates(out_dir)
    assert report == reevaluated


def test_canonical_export_proof_emits_proof_report_and_consistent_manifest(tmp_path: Path) -> None:
    out_dir = tmp_path / "canonical"
    bundle = run_canonical_live_bundle("configs/canonical_profile.yaml", artifact_dir=out_dir, export_proof=True)

    assert (out_dir / "proof_report.json").exists()
    assert bundle["proof_report_path"] == (out_dir / "proof_report.json").as_posix()
    _assert_consistent_bundle(out_dir)


def test_g5_fails_when_artifact_hash_corrupted(tmp_path: Path) -> None:
    out_dir = tmp_path / "canonical"
    run_canonical_live_bundle("configs/canonical_profile.yaml", artifact_dir=out_dir, export_proof=True)

    manifest = _load_json(out_dir / "run_manifest.json")
    manifest["artifacts"]["summary_metrics.json"] = "0" * 64
    _write_json(out_dir / "run_manifest.json", manifest)

    report = evaluate_all_gates(out_dir)
    assert report["gates"]["G5_manifest_valid"]["status"] == "FAIL"
    assert report["verdict"] == "FAIL"


def test_g5_fails_when_proof_report_hash_corrupted(tmp_path: Path) -> None:
    out_dir = tmp_path / "canonical"
    run_canonical_live_bundle("configs/canonical_profile.yaml", artifact_dir=out_dir, export_proof=True)

    manifest = _load_json(out_dir / "run_manifest.json")
    manifest["artifacts"]["proof_report.json"] = "0" * 64
    _write_json(out_dir / "run_manifest.json", manifest)

    report = evaluate_all_gates(out_dir)
    assert report["gates"]["G5_manifest_valid"]["status"] == "FAIL"


def test_g4_fails_when_required_artifact_missing_on_disk(tmp_path: Path) -> None:
    out_dir = tmp_path / "canonical"
    run_canonical_live_bundle("configs/canonical_profile.yaml", artifact_dir=out_dir, export_proof=True)

    (out_dir / "phase_space_report.json").unlink()
    report = evaluate_all_gates(out_dir)

    assert report["gates"]["G4_core_artifacts_complete"]["status"] == "FAIL"
    assert "phase_space_report.json" in report["gates"]["G4_core_artifacts_complete"]["missing_artifacts"]


def test_g4_fails_when_required_artifact_missing_in_manifest(tmp_path: Path) -> None:
    out_dir = tmp_path / "canonical"
    run_canonical_live_bundle("configs/canonical_profile.yaml", artifact_dir=out_dir, export_proof=True)

    manifest = _load_json(out_dir / "run_manifest.json")
    del manifest["artifacts"]["criticality_report.json"]
    _write_json(out_dir / "run_manifest.json", manifest)

    report = evaluate_all_gates(out_dir)
    assert report["gates"]["G4_core_artifacts_complete"]["status"] == "FAIL"
    assert "criticality_report.json" in report["gates"]["G4_core_artifacts_complete"]["missing_artifacts"]


def test_g4_fails_when_proof_report_missing_for_export_contract(tmp_path: Path) -> None:
    out_dir = tmp_path / "canonical"
    run_canonical_live_bundle("configs/canonical_profile.yaml", artifact_dir=out_dir, export_proof=True)

    (out_dir / "proof_report.json").unlink()
    report = evaluate_all_gates(out_dir)
    assert report["gates"]["G4_core_artifacts_complete"]["status"] == "FAIL"
    assert "proof_report.json" in report["gates"]["G4_core_artifacts_complete"]["missing_artifacts"]


def test_g5_fails_on_malformed_manifest_hash(tmp_path: Path) -> None:
    out_dir = tmp_path / "canonical"
    run_canonical_live_bundle("configs/canonical_profile.yaml", artifact_dir=out_dir, export_proof=True)

    manifest = _load_json(out_dir / "run_manifest.json")
    manifest["artifacts"]["summary_metrics.json"] = "xyz"
    _write_json(out_dir / "run_manifest.json", manifest)

    report = evaluate_all_gates(out_dir)
    assert report["gates"]["G5_manifest_valid"]["status"] == "FAIL"


def test_g5_fails_on_invalid_run_manifest_schema(tmp_path: Path) -> None:
    out_dir = tmp_path / "canonical"
    run_canonical_live_bundle("configs/canonical_profile.yaml", artifact_dir=out_dir, export_proof=True)

    manifest = _load_json(out_dir / "run_manifest.json")
    del manifest["schema_version"]
    _write_json(out_dir / "run_manifest.json", manifest)

    report = evaluate_all_gates(out_dir)
    assert report["gates"]["G5_manifest_valid"]["status"] == "FAIL"


def test_proof_tamper_after_finalization_causes_fail(tmp_path: Path) -> None:
    out_dir = tmp_path / "canonical"
    run_canonical_live_bundle("configs/canonical_profile.yaml", artifact_dir=out_dir, export_proof=True)

    report_path = out_dir / "proof_report.json"
    report = _load_json(report_path)
    report["failure_reasons"].append("tampered")
    _write_json(report_path, report)

    reevaluated = evaluate_all_gates(out_dir)
    assert reevaluated["gates"]["G5_manifest_valid"]["status"] == "FAIL"


def test_proof_evaluate_updates_manifest_with_proof_hash(tmp_path: Path) -> None:
    out_dir = tmp_path / "bundle"
    run_canonical_live_bundle("configs/canonical_profile.yaml", artifact_dir=out_dir, export_proof=True)

    manifest = _load_json(out_dir / "run_manifest.json")
    manifest["artifacts"]["proof_report.json"] = "0" * 64
    _write_json(out_dir / "run_manifest.json", manifest)

    proc = subprocess.run(
        [sys.executable, "-m", "bnsyn.cli", "proof-evaluate", str(out_dir)],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=_cli_env(),
    )

    assert proc.returncode == 0, proc.stderr
    _assert_consistent_bundle(out_dir)


def test_proof_report_is_deterministic_across_repeated_runs(tmp_path: Path) -> None:
    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"

    run_canonical_live_bundle("configs/canonical_profile.yaml", artifact_dir=out_a, export_proof=True)
    run_canonical_live_bundle("configs/canonical_profile.yaml", artifact_dir=out_b, export_proof=True)

    for filename in [
        "summary_metrics.json",
        "criticality_report.json",
        "avalanche_report.json",
        "phase_space_report.json",
        "population_rate_trace.npy",
        "sigma_trace.npy",
        "coherence_trace.npy",
        "phase_space_rate_sigma.png",
        "phase_space_rate_coherence.png",
        "phase_space_activity_map.png",
        "avalanche_fit_report.json",
        "robustness_report.json",
        "envelope_report.json",
        "proof_report.json",
        "emergence_plot.png",
        "raster_plot.png",
        "population_rate_plot.png",
        "run_manifest.json",
    ]:
        assert (out_a / filename).read_bytes() == (out_b / filename).read_bytes()


def test_final_proof_report_evaluates_against_final_manifest_state(tmp_path: Path) -> None:
    out_dir = tmp_path / "canonical"
    run_canonical_live_bundle("configs/canonical_profile.yaml", artifact_dir=out_dir, export_proof=True)
    _assert_consistent_bundle(out_dir)


def test_no_stale_hash_after_export_proof_run(tmp_path: Path) -> None:
    out_dir = tmp_path / "canonical"
    run_canonical_live_bundle("configs/canonical_profile.yaml", artifact_dir=out_dir, export_proof=True)
    _assert_consistent_bundle(out_dir)


def test_registry_alignment_for_required_artifacts(tmp_path: Path) -> None:
    out_dir = tmp_path / "canonical"
    run_canonical_live_bundle("configs/canonical_profile.yaml", artifact_dir=out_dir, export_proof=True)

    registry = _load_json(ROOT / "ci" / "validation_gates.json")
    by_id = {gate["id"]: gate for gate in registry["registry"]}
    required = set(by_id["G4_core_artifacts_complete"]["threshold"]["required_artifacts_by_mode"]["canonical-export-proof"])
    report = _load_json(out_dir / "proof_report.json")

    assert required == set(report["artifacts_verified"])


def test_fail_closed_on_malformed_registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    malformed_registry = tmp_path / "bad_registry.json"
    _write_json(malformed_registry, {"schema_version": "1", "registry": [{"id": "G1_active_spiking"}]})
    monkeypatch.setattr(proof_evaluate, "VALIDATION_GATES_PATH", malformed_registry)

    out_dir = tmp_path / "canonical"
    run_canonical_live_bundle("configs/canonical_profile.yaml", artifact_dir=out_dir)
    report = evaluate_all_gates(out_dir)

    assert report["verdict"] == "FAIL"
    assert any("fail-closed" in reason for reason in report["failure_reasons"])


def test_fail_closed_on_malformed_run_manifest_schema(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    malformed_schema = tmp_path / "bad_manifest_schema.json"
    malformed_schema.write_text('{"type": 7}', encoding="utf-8")
    monkeypatch.setattr(proof_evaluate, "RUN_MANIFEST_SCHEMA_PATH", malformed_schema)

    out_dir = tmp_path / "canonical"
    run_canonical_live_bundle("configs/canonical_profile.yaml", artifact_dir=out_dir, export_proof=True)
    report = evaluate_all_gates(out_dir)

    assert report["gates"]["G5_manifest_valid"]["status"] == "FAIL"


def test_export_manifest_tamper_cmd_to_base_fails_g5(tmp_path: Path) -> None:
    out_dir = tmp_path / "canonical"
    run_canonical_live_bundle("configs/canonical_profile.yaml", artifact_dir=out_dir, export_proof=True)
    manifest = _load_json(out_dir / "run_manifest.json")
    manifest["cmd"] = "bnsyn run --profile canonical --plot"
    _write_json(out_dir / "run_manifest.json", manifest)
    report = evaluate_all_gates(out_dir)
    assert report["gates"]["G5_manifest_valid"]["status"] == "FAIL"


def test_export_manifest_tamper_contract_to_base_fails_g5(tmp_path: Path) -> None:
    out_dir = tmp_path / "canonical"
    run_canonical_live_bundle("configs/canonical_profile.yaml", artifact_dir=out_dir, export_proof=True)
    manifest = _load_json(out_dir / "run_manifest.json")
    manifest["bundle_contract"] = "canonical-base"
    _write_json(out_dir / "run_manifest.json", manifest)
    report = evaluate_all_gates(out_dir)
    assert report["gates"]["G5_manifest_valid"]["status"] == "FAIL"


def test_export_manifest_tamper_export_proof_false_fails_g5(tmp_path: Path) -> None:
    out_dir = tmp_path / "canonical"
    run_canonical_live_bundle("configs/canonical_profile.yaml", artifact_dir=out_dir, export_proof=True)
    manifest = _load_json(out_dir / "run_manifest.json")
    manifest["export_proof"] = False
    _write_json(out_dir / "run_manifest.json", manifest)
    report = evaluate_all_gates(out_dir)
    assert report["gates"]["G5_manifest_valid"]["status"] == "FAIL"


def test_base_manifest_tamper_add_proof_entry_fails_g5(tmp_path: Path) -> None:
    out_dir = tmp_path / "canonical"
    run_canonical_live_bundle("configs/canonical_profile.yaml", artifact_dir=out_dir, export_proof=False)
    manifest = _load_json(out_dir / "run_manifest.json")
    manifest["artifacts"]["proof_report.json"] = "0" * 64
    _write_json(out_dir / "run_manifest.json", manifest)
    report = evaluate_all_gates(out_dir)
    assert report["gates"]["G5_manifest_valid"]["status"] == "FAIL"


def test_proof_report_mode_mismatch_vs_manifest_fails_g5(tmp_path: Path) -> None:
    out_dir = tmp_path / "canonical"
    run_canonical_live_bundle("configs/canonical_profile.yaml", artifact_dir=out_dir, export_proof=True)
    proof = _load_json(out_dir / "proof_report.json")
    proof["bundle_contract"] = "canonical-base"
    _write_json(out_dir / "proof_report.json", proof)
    report = evaluate_all_gates(out_dir)
    assert report["gates"]["G5_manifest_valid"]["status"] == "FAIL"


def test_cli_canonical_run_artifact_list_without_export_proof(tmp_path: Path) -> None:
    out_dir = tmp_path / "cli_base"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "bnsyn.cli",
            "run",
            "--profile",
            "canonical",
            "--plot",
            "--output",
            str(out_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=_cli_env(),
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["bundle_contract"] == "canonical-base"
    assert payload["artifacts"] == [
        "emergence_plot.png",
        "summary_metrics.json",
        "criticality_report.json",
        "avalanche_report.json",
        "phase_space_report.json",
        "population_rate_trace.npy",
        "sigma_trace.npy",
        "coherence_trace.npy",
        "phase_space_rate_sigma.png",
        "phase_space_rate_coherence.png",
        "phase_space_activity_map.png",
        "avalanche_fit_report.json",
        "robustness_report.json",
        "envelope_report.json",
        "run_manifest.json",
    ]


def test_cli_canonical_export_proof_internal_consistency(tmp_path: Path) -> None:
    out_dir = tmp_path / "cli_canonical"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "bnsyn.cli",
            "run",
            "--profile",
            "canonical",
            "--plot",
            "--export-proof",
            "--output",
            str(out_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=_cli_env(),
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["bundle_contract"] == "canonical-export-proof"
    assert payload["artifacts"] == [
        "emergence_plot.png",
        "summary_metrics.json",
        "criticality_report.json",
        "avalanche_report.json",
        "phase_space_report.json",
        "population_rate_trace.npy",
        "sigma_trace.npy",
        "coherence_trace.npy",
        "phase_space_rate_sigma.png",
        "phase_space_rate_coherence.png",
        "phase_space_activity_map.png",
        "avalanche_fit_report.json",
        "robustness_report.json",
        "envelope_report.json",
        "run_manifest.json",
        "proof_report.json",
    ]
    _assert_consistent_bundle(out_dir)


def test_proof_validate_bundle_fails_when_envelope_report_missing(tmp_path: Path) -> None:
    from bnsyn.proof.bundle_validator import validate_canonical_bundle

    out_dir = tmp_path / "canonical"
    run_canonical_live_bundle("configs/canonical_profile.yaml", artifact_dir=out_dir, export_proof=True)
    (out_dir / "envelope_report.json").unlink()

    result = validate_canonical_bundle(out_dir)
    assert result["status"] == "FAIL"
    assert any("missing artifact: envelope_report.json" in err for err in result["errors"])


def test_proof_validate_bundle_schema_error_reports_json_path(tmp_path: Path) -> None:
    from bnsyn.proof.bundle_validator import validate_canonical_bundle

    out_dir = tmp_path / "canonical"
    run_canonical_live_bundle("configs/canonical_profile.yaml", artifact_dir=out_dir, export_proof=True)

    envelope_path = out_dir / "envelope_report.json"
    envelope = _load_json(envelope_path)
    del envelope["verdict"]
    _write_json(envelope_path, envelope)

    result = validate_canonical_bundle(out_dir)
    assert result["status"] == "FAIL"
    assert any("json_path" in err or "$" in err for err in result["errors"])
