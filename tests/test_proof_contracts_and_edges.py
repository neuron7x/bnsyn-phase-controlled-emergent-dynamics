from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bnsyn.experiments.declarative import run_canonical_live_bundle
from bnsyn.proof import evaluate as proof_evaluate
from bnsyn.proof.contracts import (
    BASE_ARTIFACTS,
    CANONICAL_BASE_COMMAND,
    CANONICAL_BASE_CONTRACT,
    CANONICAL_EXPORT_PROOF_COMMAND,
    CANONICAL_EXPORT_PROOF_CONTRACT,
    EXPORT_PROOF_ARTIFACTS,
    ManifestMode,
    artifacts_for_export_proof,
    bundle_contract_for_export_proof,
    command_for_export_proof,
    mode_from_manifest,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contract_helpers_select_mode_specific_constants() -> None:
    assert command_for_export_proof(False) == CANONICAL_BASE_COMMAND
    assert command_for_export_proof(True) == CANONICAL_EXPORT_PROOF_COMMAND
    assert bundle_contract_for_export_proof(False) == CANONICAL_BASE_CONTRACT
    assert bundle_contract_for_export_proof(True) == CANONICAL_EXPORT_PROOF_CONTRACT
    assert artifacts_for_export_proof(False) == BASE_ARTIFACTS
    assert artifacts_for_export_proof(True) == EXPORT_PROOF_ARTIFACTS


def test_mode_from_manifest_validates_types_and_mode_rules() -> None:
    mode, errors = mode_from_manifest({"cmd": 1, "bundle_contract": None, "export_proof": "x", "artifacts": []})
    assert mode is None
    assert "manifest cmd must be string" in errors
    assert "manifest bundle_contract invalid" in errors
    assert "manifest export_proof must be boolean" in errors
    assert "manifest artifacts must be object" in errors

    export_manifest = {
        "cmd": CANONICAL_EXPORT_PROOF_COMMAND,
        "bundle_contract": CANONICAL_EXPORT_PROOF_CONTRACT,
        "export_proof": True,
        "artifacts": {"proof_report.json": "0" * 64},
    }
    mode, errors = mode_from_manifest(export_manifest)
    assert errors == []
    assert mode == ManifestMode(
        bundle_contract=CANONICAL_EXPORT_PROOF_CONTRACT,
        export_proof=True,
        cmd=CANONICAL_EXPORT_PROOF_COMMAND,
    )

    base_manifest = {
        "cmd": CANONICAL_BASE_COMMAND,
        "bundle_contract": CANONICAL_BASE_CONTRACT,
        "export_proof": False,
        "artifacts": {"proof_report.json": "0" * 64},
    }
    mode, errors = mode_from_manifest(base_manifest)
    assert mode is None
    assert "base mode forbids proof_report.json manifest entry" in errors


def test_load_json_and_registry_parsing_fail_closed_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    not_obj = tmp_path / "not_obj.json"
    not_obj.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected JSON object"):
        proof_evaluate._load_json(not_obj)

    missing_registry = tmp_path / "missing_registry.json"
    _write_json(missing_registry, {"schema_version": "1.0.0"})
    monkeypatch.setattr(proof_evaluate, "VALIDATION_GATES_PATH", missing_registry)
    with pytest.raises(ValueError, match="validation gate registry missing"):
        proof_evaluate._load_gate_registry()

    malformed_registry = tmp_path / "malformed_registry.json"
    _write_json(malformed_registry, {"registry": [{"id": 3}]})
    monkeypatch.setattr(proof_evaluate, "VALIDATION_GATES_PATH", malformed_registry)
    with pytest.raises(ValueError, match="malformed gate registry entry"):
        proof_evaluate._load_gate_registry()


def test_required_artifacts_from_registry_fail_closed_paths() -> None:
    mode = ManifestMode(bundle_contract=CANONICAL_BASE_CONTRACT, export_proof=False, cmd=CANONICAL_BASE_COMMAND)
    with pytest.raises(ValueError, match="G4 threshold missing"):
        proof_evaluate._required_artifacts_from_registry({"G4_core_artifacts_complete": {}}, mode)

    with pytest.raises(ValueError, match="G4 required_artifacts_by_mode missing"):
        proof_evaluate._required_artifacts_from_registry(
            {"G4_core_artifacts_complete": {"threshold": {}}},
            mode,
        )

    with pytest.raises(ValueError, match="invalid for canonical-base"):
        proof_evaluate._required_artifacts_from_registry(
            {"G4_core_artifacts_complete": {"threshold": {"required_artifacts_by_mode": {"canonical-base": [1]}}}},
            mode,
        )

    with pytest.raises(ValueError, match="registry/runtime artifact contract drift"):
        proof_evaluate._required_artifacts_from_registry(
            {
                "G4_core_artifacts_complete": {
                    "threshold": {"required_artifacts_by_mode": {"canonical-base": ["summary_metrics.json"]}}
                }
            },
            mode,
        )


def test_numeric_gate_validation_failures() -> None:
    with pytest.raises(ValueError, match="numeric gate threshold missing"):
        proof_evaluate._evaluate_numeric_gate({}, {})
    with pytest.raises(ValueError, match="numeric gate metric missing"):
        proof_evaluate._evaluate_numeric_gate({"threshold": {"op": ">", "value": 1}}, {})
    with pytest.raises(ValueError, match="metric spike_events missing"):
        proof_evaluate._evaluate_numeric_gate({"threshold": {"metric": "spike_events", "op": ">", "value": 1}}, {})
    with pytest.raises(ValueError, match="numeric gate value missing"):
        proof_evaluate._evaluate_numeric_gate({"threshold": {"metric": "spike_events", "op": ">", "value": "x"}}, {"spike_events": 2})
    with pytest.raises(ValueError, match="between gate requires"):
        proof_evaluate._evaluate_numeric_gate(
            {"threshold": {"metric": "spike_events", "op": "between", "value": [1]}},
            {"spike_events": 2},
        )
    with pytest.raises(ValueError, match="unsupported gate op"):
        proof_evaluate._evaluate_numeric_gate(
            {"threshold": {"metric": "spike_events", "op": "<", "value": 3}},
            {"spike_events": 2},
        )


def test_g4_and_g5_error_paths_and_unreadable_proof_report(tmp_path: Path) -> None:
    out_dir = tmp_path / "bundle"
    run_canonical_live_bundle("configs/canonical_profile.yaml", artifact_dir=out_dir, export_proof=True)

    manifest = _load_json(out_dir / "run_manifest.json")
    g4, verified = proof_evaluate.evaluate_gate_g4_artifact_contract(
        out_dir,
        {"artifacts": []},
        [],
        BASE_ARTIFACTS,
    )
    assert g4["status"] == "FAIL"
    assert verified == []

    g4, verified = proof_evaluate.evaluate_gate_g4_artifact_contract(out_dir, manifest, ["mode bad"], BASE_ARTIFACTS)
    assert g4["status"] == "FAIL"
    assert verified == []

    # unreadable proof report should be fail-closed inside G5 validation details
    (out_dir / "proof_report.json").write_text("{broken-json", encoding="utf-8")
    g5 = proof_evaluate.evaluate_gate_g5_manifest_valid(out_dir, manifest, [])
    assert g5["status"] == "FAIL"
    assert any("proof_report.json unreadable" in e for e in g5["errors"])

    manifest_bad_self = dict(manifest)
    manifest_bad_self["artifacts"] = dict(manifest["artifacts"])
    manifest_bad_self["artifacts"]["run_manifest.json"] = "not-self"
    g5 = proof_evaluate.evaluate_gate_g5_manifest_valid(out_dir, manifest_bad_self, [])
    assert g5["status"] == "FAIL"
    assert "run_manifest.json entry must be self-unhashed" in g5["errors"]

    manifest_bad_types = dict(manifest)
    manifest_bad_types["artifacts"] = {"summary_metrics.json": 1}
    g5 = proof_evaluate.evaluate_gate_g5_manifest_valid(out_dir, manifest_bad_types, [])
    assert g5["status"] == "FAIL"
    assert "artifact entries must be string:string" in g5["errors"]

    g5 = proof_evaluate.evaluate_gate_g5_manifest_valid(out_dir, {"artifacts": []}, [])
    assert g5["status"] == "FAIL"
    assert "manifest artifacts is not an object" in g5["errors"]


def test_misc_helpers_and_discover_seed_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid gate status"):
        proof_evaluate._gate_state({"G1": {"status": "bad"}}, "G1")

    verdict, code, reasons = proof_evaluate._compute_verdict(
        {"G1": {"status": "PASS"}},
        {"G1": {"status": "wired"}},
    )
    assert (verdict, code, reasons) == ("PASS", 0, [])

    artifact_dir = tmp_path / "seed"
    artifact_dir.mkdir()
    assert proof_evaluate._discover_seed(artifact_dir) == 0
    _write_json(artifact_dir / "summary_metrics.json", {"seed": -1})
    assert proof_evaluate._discover_seed(artifact_dir) == 0
    _write_json(artifact_dir / "summary_metrics.json", {"seed": 7})
    assert proof_evaluate._discover_seed(artifact_dir) == 7


def test_evaluate_and_emit_non_export_branch_writes_report(tmp_path: Path) -> None:
    out_dir = tmp_path / "base"
    run_canonical_live_bundle("configs/canonical_profile.yaml", artifact_dir=out_dir, export_proof=False)
    result = proof_evaluate.evaluate_and_emit(out_dir)
    assert result.report_path == out_dir / "proof_report.json"
    assert result.report["bundle_contract"] == "canonical-base"
    assert (out_dir / "proof_report.json").exists()


def test_update_manifest_proof_hash_requires_artifacts_object(tmp_path: Path) -> None:
    out_dir = tmp_path / "artifact"
    out_dir.mkdir()
    _write_json(out_dir / "run_manifest.json", {"artifacts": []})
    with pytest.raises(ValueError, match="run_manifest artifacts must be object"):
        proof_evaluate._update_manifest_proof_hash(out_dir, "0" * 64)


def test_evaluate_and_emit_stabilizes_on_third_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_dir = tmp_path / "bundle"
    out_dir.mkdir()
    _write_json(out_dir / "run_manifest.json", {"export_proof": True, "artifacts": {"run_manifest.json": "self-unhashed"}})

    reports = iter(
        [
            {"k": "first"},
            {"k": "second"},
            {"k": "third"},
            {"k": "third"},
        ]
    )

    def fake_eval(_: Path) -> dict:
        return next(reports)

    monkeypatch.setattr(proof_evaluate, "evaluate_all_gates", fake_eval)
    monkeypatch.setattr(proof_evaluate, "emit_proof_report", lambda result, artifact_dir: Path(artifact_dir) / "proof_report.json")
    monkeypatch.setattr(proof_evaluate, "sha256_file", lambda _: "a" * 64)
    monkeypatch.setattr(proof_evaluate, "_update_manifest_proof_hash", lambda *_args: None)

    result = proof_evaluate.evaluate_and_emit(out_dir)
    assert result.report == {"k": "third"}
    assert result.report_path == out_dir / "proof_report.json"


def test_evaluate_and_emit_raises_if_not_stable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_dir = tmp_path / "bundle"
    out_dir.mkdir()
    _write_json(out_dir / "run_manifest.json", {"export_proof": True, "artifacts": {"run_manifest.json": "self-unhashed"}})

    reports = iter(
        [
            {"k": "first"},
            {"k": "second"},
            {"k": "third"},
            {"k": "fourth"},
        ]
    )

    monkeypatch.setattr(proof_evaluate, "evaluate_all_gates", lambda _: next(reports))
    monkeypatch.setattr(proof_evaluate, "emit_proof_report", lambda result, artifact_dir: Path(artifact_dir) / "proof_report.json")
    monkeypatch.setattr(proof_evaluate, "sha256_file", lambda _: "a" * 64)
    monkeypatch.setattr(proof_evaluate, "_update_manifest_proof_hash", lambda *_args: None)

    with pytest.raises(RuntimeError, match="failed to stabilize"):
        proof_evaluate.evaluate_and_emit(out_dir)


def test_load_numeric_trace_validation_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = tmp_path / "missing.npy"
    with pytest.raises(ValueError, match="missing required trace artifact"):
        proof_evaluate._load_numeric_trace(missing, metric_name="rate_mean_hz")

    arr_path = tmp_path / "trace.npy"
    np.save(arr_path, np.asarray([], dtype=float))
    with pytest.raises(ValueError, match="is empty"):
        proof_evaluate._load_numeric_trace(arr_path, metric_name="rate_mean_hz")

    np.save(arr_path, np.asarray([1.0, np.nan], dtype=float))
    with pytest.raises(ValueError, match="contains non-finite"):
        proof_evaluate._load_numeric_trace(arr_path, metric_name="rate_mean_hz")

    monkeypatch.setattr(proof_evaluate.np, "load", lambda _: [1.0, 2.0])
    with pytest.raises(ValueError, match="is not a numpy array"):
        proof_evaluate._load_numeric_trace(arr_path, metric_name="rate_mean_hz")


def test_extract_spike_events_from_raw_npz_edge_paths(tmp_path: Path) -> None:
    bad_npz = tmp_path / "traces.npz"
    np.savez(bad_npz, spike_steps=np.asarray([[1, 2]], dtype=np.int64))
    assert proof_evaluate._extract_spike_events_from_raw_npz(tmp_path) is None

    good_npz = tmp_path / "good.npz"
    np.savez(good_npz, spike_steps=np.asarray([1, 2, 3], dtype=np.int64), spike_neurons=np.asarray([0, 1, 2], dtype=np.int64))
    events = proof_evaluate._extract_spike_events_from_raw_npz(tmp_path)
    assert events == (3, "good.npz")


def test_recompute_metrics_from_artifacts_unverifiable_paths(tmp_path: Path) -> None:
    np.save(tmp_path / "population_rate_trace.npy", np.asarray([0.0, 0.0], dtype=float))
    np.save(tmp_path / "sigma_trace.npy", np.asarray([1.0, 1.0], dtype=float))

    no_meta = proof_evaluate.recompute_metrics_from_artifacts(tmp_path, summary={}, manifest={})
    assert any("missing dt_ms or N" in err for err in no_meta["errors"])

    non_positive = proof_evaluate.recompute_metrics_from_artifacts(
        tmp_path,
        summary={"dt_ms": 0.1, "N": 100},
        manifest={"dt_ms": -1.0, "N": 100},
    )
    assert any("non-positive dt_ms or N" in err for err in non_positive["errors"])


def test_recompute_metrics_from_artifacts_rate_reconstruction_tolerance_failure(tmp_path: Path) -> None:
    np.save(tmp_path / "population_rate_trace.npy", np.asarray([1.23456789], dtype=float))
    np.save(tmp_path / "sigma_trace.npy", np.asarray([1.0], dtype=float))
    recomputed = proof_evaluate.recompute_metrics_from_artifacts(
        tmp_path,
        summary={"dt_ms": 1.0, "N": 1},
        manifest={"dt_ms": 1.0, "N": 1},
    )
    assert any("non-integer reconstruction" in err for err in recomputed["errors"])


def test_metric_consistency_gate_reports_missing_metrics() -> None:
    result = proof_evaluate.evaluate_gate_g9_metric_consistency(
        summary={"rate_mean_hz": 1.0},
        recomputed={"metrics": {"rate_mean_hz": 1.0, "sigma_mean": 1.0}, "errors": []},
    )
    assert result["status"] == "FAIL"
    assert "spike_events: missing summary metric" in result["errors"]


def test_manifest_float_prefers_manifest_then_summary() -> None:
    assert proof_evaluate._manifest_float({"dt_ms": 0.5}, {"dt_ms": 0.1}, "dt_ms") == 0.5
    assert proof_evaluate._manifest_float({}, {"dt_ms": 0.1}, "dt_ms") == 0.1
    assert proof_evaluate._manifest_float({}, {}, "dt_ms") is None
