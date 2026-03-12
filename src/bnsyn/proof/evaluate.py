from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema  # type: ignore[import-untyped]
import numpy as np

from .contracts import (
    CANONICAL_BASE_CONTRACT,
    ManifestMode,
    mode_from_manifest,
)

ROOT = Path(__file__).resolve().parents[3]
PROOF_SCHEMA_PATH = ROOT / "schemas" / "proof-report.schema.json"
RUN_MANIFEST_SCHEMA_PATH = ROOT / "schemas" / "run-manifest.schema.json"
VALIDATION_GATES_PATH = ROOT / "ci" / "validation_gates.json"
PROOF_SCHEMA_VERSION = "1.0.0"
DETERMINISTIC_TIMESTAMP_UTC = "1970-01-01T00:00:00Z"
EXPECTED_GATE_IDS = (
    "G1_active_spiking",
    "G2_rate_in_bounds",
    "G3_sigma_in_range",
    "G4_core_artifacts_complete",
    "G5_manifest_valid",
    "G6_determinism_replay",
    "G7_avalanche_evidence_sufficient",
    "G8_reproducibility_envelope",
    "G9_metrics_trace_consistency",
)

RATE_MEAN_TOLERANCE = 1e-12
SIGMA_MEAN_TOLERANCE = 1e-12
SPIKE_EVENTS_FALLBACK_ROUNDING_TOLERANCE = 1e-6

# G4 required-artifact floor remains registry-driven and intentionally narrower
# than canonical CLI payload artifacts, which may include additive evidence files.
G4_BASE_REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "emergence_plot.png",
    "summary_metrics.json",
    "criticality_report.json",
    "avalanche_report.json",
    "phase_space_report.json",
    "avalanche_fit_report.json",
    "robustness_report.json",
    "envelope_report.json",
    "run_manifest.json",
)
G4_EXPORT_REQUIRED_ARTIFACTS: tuple[str, ...] = G4_BASE_REQUIRED_ARTIFACTS + ("proof_report.json",)


@dataclass(frozen=True)
class EvaluationResult:
    report: dict[str, Any]
    report_path: Path


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"Expected JSON object in {path}"
        raise ValueError(msg)
    return payload


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_gate_registry() -> dict[str, dict[str, Any]]:
    payload = _load_json(VALIDATION_GATES_PATH)
    registry_raw = payload.get("registry")
    if not isinstance(registry_raw, list):
        raise ValueError("validation gate registry missing")
    registry: dict[str, dict[str, Any]] = {}
    for gate in registry_raw:
        if not isinstance(gate, dict) or not isinstance(gate.get("id"), str):
            raise ValueError("malformed gate registry entry")
        registry[gate["id"]] = gate

    missing = [gate_id for gate_id in EXPECTED_GATE_IDS if gate_id not in registry]
    if missing:
        raise ValueError(f"missing expected gates: {missing}")
    return registry


def _required_artifacts_from_registry(registry: dict[str, dict[str, Any]], mode: ManifestMode) -> tuple[str, ...]:
    g4 = registry["G4_core_artifacts_complete"]
    threshold = g4.get("threshold")
    if not isinstance(threshold, dict):
        raise ValueError("G4 threshold missing")
    required_by_mode = threshold.get("required_artifacts_by_mode")
    if not isinstance(required_by_mode, dict):
        raise ValueError("G4 required_artifacts_by_mode missing")
    mode_required = required_by_mode.get(mode.bundle_contract)
    if not isinstance(mode_required, list) or not all(isinstance(x, str) and x for x in mode_required):
        raise ValueError(f"G4 required_artifacts_by_mode invalid for {mode.bundle_contract}")

    required_artifacts = tuple(mode_required)
    expected_floor = G4_EXPORT_REQUIRED_ARTIFACTS if mode.export_proof else G4_BASE_REQUIRED_ARTIFACTS
    if set(required_artifacts) != set(expected_floor):
        raise ValueError("registry/runtime artifact contract drift")
    return required_artifacts


def load_artifacts(artifact_dir: str | Path) -> dict[str, Any]:
    root = Path(artifact_dir)
    summary = _load_json(root / "summary_metrics.json")
    manifest = _load_json(root / "run_manifest.json")
    registry = _load_gate_registry()
    return {"artifact_dir": root, "summary": summary, "manifest": manifest, "registry": registry}


def _load_numeric_trace(path: Path, *, metric_name: str) -> np.ndarray:
    if not path.is_file():
        raise ValueError(f"missing required trace artifact: {path.name}")
    trace = np.load(path)
    if not isinstance(trace, np.ndarray):
        raise ValueError(f"{path.name} is not a numpy array")
    values = np.asarray(trace, dtype=np.float64)
    if values.size == 0:
        raise ValueError(f"{path.name} is empty")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path.name} contains non-finite values")
    return values


def _extract_spike_events_from_raw_npz(artifact_dir: Path) -> tuple[int, str] | None:
    candidates = [artifact_dir / "traces.npz", *sorted(artifact_dir.glob("*.npz"))]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        try:
            with np.load(candidate, allow_pickle=False) as payload:
                if "spike_steps" in payload.files:
                    spike_steps = np.asarray(payload["spike_steps"], dtype=np.int64)
                    if spike_steps.ndim != 1:
                        raise ValueError(f"{candidate.name}: spike_steps must be 1D")
                    if not np.all(np.isfinite(spike_steps)):
                        raise ValueError(f"{candidate.name}: spike_steps contains non-finite values")
                    if "spike_neurons" in payload.files:
                        spike_neurons = np.asarray(payload["spike_neurons"], dtype=np.int64)
                        if spike_neurons.ndim != 1:
                            raise ValueError(f"{candidate.name}: spike_neurons must be 1D")
                        if spike_neurons.size != spike_steps.size:
                            raise ValueError(f"{candidate.name}: spike_neurons length mismatch")
                    return int(spike_steps.size), candidate.name
        except Exception:
            continue
    return None


def _manifest_float(manifest: dict[str, Any], summary: dict[str, Any], key: str) -> float | None:
    value = manifest.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    fallback = summary.get(key)
    if isinstance(fallback, (int, float)):
        return float(fallback)
    return None


def recompute_metrics_from_artifacts(artifact_dir: Path, summary: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "metrics": {},
        "sources": {},
        "errors": [],
        "spike_events_source": "unverifiable",
    }

    try:
        rate_trace = _load_numeric_trace(artifact_dir / "population_rate_trace.npy", metric_name="rate_mean_hz")
        result["metrics"]["rate_mean_hz"] = float(np.mean(rate_trace))
        result["sources"]["rate_mean_hz"] = "population_rate_trace.npy"
    except ValueError as exc:
        result["errors"].append(str(exc))

    try:
        sigma_trace = _load_numeric_trace(artifact_dir / "sigma_trace.npy", metric_name="sigma_mean")
        result["metrics"]["sigma_mean"] = float(np.mean(sigma_trace))
        result["sources"]["sigma_mean"] = "sigma_trace.npy"
    except ValueError as exc:
        result["errors"].append(str(exc))

    raw_spike = _extract_spike_events_from_raw_npz(artifact_dir)
    if raw_spike is not None:
        spike_events, source = raw_spike
        result["metrics"]["spike_events"] = spike_events
        result["sources"]["spike_events"] = source
        result["spike_events_source"] = "raw_npz"
        return result

    dt_ms = _manifest_float(manifest, summary, "dt_ms")
    n_neurons = _manifest_float(manifest, summary, "N")
    if dt_ms is None or n_neurons is None:
        result["errors"].append("spike_events unverifiable: missing dt_ms or N for deterministic rate-trace reconstruction")
        return result
    if dt_ms <= 0 or n_neurons <= 0:
        result["errors"].append("spike_events unverifiable: non-positive dt_ms or N")
        return result

    if "rate_mean_hz" not in result["metrics"]:
        result["errors"].append("spike_events unverifiable: population_rate_trace.npy unavailable")
        return result

    rate_trace = _load_numeric_trace(artifact_dir / "population_rate_trace.npy", metric_name="spike_events")
    per_step = rate_trace * n_neurons * (dt_ms / 1000.0)
    rounded = np.rint(per_step)
    max_delta = float(np.max(np.abs(per_step - rounded))) if per_step.size else 0.0
    if max_delta > SPIKE_EVENTS_FALLBACK_ROUNDING_TOLERANCE:
        result["errors"].append(
            "spike_events unverifiable: non-integer reconstruction from population_rate_trace.npy exceeds tolerance"
        )
        return result

    reconstructed = int(np.sum(rounded.astype(np.int64)))
    result["metrics"]["spike_events"] = reconstructed
    result["sources"]["spike_events"] = "population_rate_trace.npy"
    result["spike_events_source"] = "rate_trace_reconstruction"
    result["spike_events_reconstruction"] = {
        "dt_ms": dt_ms,
        "N": n_neurons,
        "rounding_tolerance": SPIKE_EVENTS_FALLBACK_ROUNDING_TOLERANCE,
        "max_rounding_delta": max_delta,
    }
    return result


def _evaluate_numeric_gate_from_value(metric_name: str, metric_value: float, gate: dict[str, Any]) -> dict[str, Any]:
    threshold = gate.get("threshold")
    if not isinstance(threshold, dict):
        raise ValueError("numeric gate threshold missing")
    op = threshold.get("op")
    value = threshold.get("value")

    if op == ">":
        if not isinstance(value, (int, float)):
            raise ValueError("numeric gate value missing")
        return {
            "status": "PASS" if metric_value > float(value) else "FAIL",
            "value": metric_value,
            "details": f"{metric_name} > {value}",
        }

    if op == "between":
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError("between gate requires [min,max]")
        lower = float(value[0])
        upper = float(value[1])
        return {
            "status": "PASS" if lower <= metric_value <= upper else "FAIL",
            "value": metric_value,
            "details": f"{lower} <= {metric_name} <= {upper}",
        }

    raise ValueError(f"unsupported gate op: {op}")


def evaluate_gate_g1_active_spiking(metrics: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    metric_name = str(gate.get("threshold", {}).get("metric", "spike_events"))
    if metric_name not in metrics:
        return {"status": "FAIL", "details": f"recomputed metric unavailable: {metric_name}"}
    return _evaluate_numeric_gate_from_value(metric_name, float(metrics[metric_name]), gate)


def evaluate_gate_g2_rate_bounds(metrics: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    metric_name = str(gate.get("threshold", {}).get("metric", "rate_mean_hz"))
    if metric_name not in metrics:
        return {"status": "FAIL", "details": f"recomputed metric unavailable: {metric_name}"}
    return _evaluate_numeric_gate_from_value(metric_name, float(metrics[metric_name]), gate)


def evaluate_gate_g3_sigma_range(metrics: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    metric_name = str(gate.get("threshold", {}).get("metric", "sigma_mean"))
    if metric_name not in metrics:
        return {"status": "FAIL", "details": f"recomputed metric unavailable: {metric_name}"}
    return _evaluate_numeric_gate_from_value(metric_name, float(metrics[metric_name]), gate)


def evaluate_gate_g9_metric_consistency(summary: dict[str, Any], recomputed: dict[str, Any]) -> dict[str, Any]:
    recomputed_metrics = recomputed.get("metrics", {})
    if not isinstance(recomputed_metrics, dict):
        return {"status": "FAIL", "details": "invalid recomputed metrics payload"}
    recompute_errors = recomputed.get("errors", [])
    if isinstance(recompute_errors, list) and recompute_errors:
        return {
            "status": "FAIL",
            "details": "trace metric recompute failed",
            "errors": [str(err) for err in recompute_errors],
        }

    spike_policy = "exact_raw_npz" if recomputed.get("spike_events_source") == "raw_npz" else "exact_rate_trace_reconstruction"
    comparisons: dict[str, dict[str, Any]] = {
        "spike_events": {"tolerance": 0.0, "policy": spike_policy},
        "rate_mean_hz": {"tolerance": RATE_MEAN_TOLERANCE, "policy": "abs_delta <= tolerance"},
        "sigma_mean": {"tolerance": SIGMA_MEAN_TOLERANCE, "policy": "abs_delta <= tolerance"},
    }

    errors: list[str] = []
    for name, policy in comparisons.items():
        expected = summary.get(name)
        actual = recomputed_metrics.get(name)
        if expected is None:
            policy.update({"status": "FAIL", "reason": "missing summary metric"})
            errors.append(f"{name}: missing summary metric")
            continue
        if actual is None:
            policy.update({"status": "FAIL", "reason": "missing recomputed metric"})
            errors.append(f"{name}: missing recomputed metric")
            continue

        expected_f = float(expected)
        actual_f = float(actual)
        delta = abs(actual_f - expected_f)
        policy.update({"summary": expected, "recomputed": actual, "delta": delta})
        if delta <= float(policy["tolerance"]):
            policy["status"] = "PASS"
        else:
            policy["status"] = "FAIL"
            policy["reason"] = "delta exceeds tolerance"
            errors.append(f"{name}: delta {delta} exceeds tolerance {policy['tolerance']}")

    details = "summary metrics align with trace recompute" if not errors else "summary metrics inconsistent with trace recompute"
    return {
        "status": "PASS" if not errors else "FAIL",
        "details": details,
        "consistency": comparisons,
        **({"errors": errors} if errors else {}),
    }


def evaluate_gate_g4_artifact_contract(
    artifact_dir: Path,
    manifest: dict[str, Any],
    mode_errors: list[str],
    required_artifacts: tuple[str, ...],
) -> tuple[dict[str, Any], list[str]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        return ({"status": "FAIL", "details": "manifest artifacts must be object", "missing_artifacts": list(required_artifacts)}, [])

    if mode_errors:
        return ({"status": "FAIL", "details": "manifest mode invalid", "missing_artifacts": list(required_artifacts)}, [])

    missing_in_manifest = [name for name in required_artifacts if name not in artifacts]
    missing_on_disk = [name for name in required_artifacts if not (artifact_dir / name).is_file()]
    missing = sorted(set(missing_in_manifest + missing_on_disk))
    verified = [name for name in required_artifacts if name not in missing and (artifact_dir / name).is_file()]

    status = "PASS" if not missing else "FAIL"
    return ({"status": status, "details": "required artifacts present in manifest and on disk", "missing_artifacts": missing}, verified)


def _validate_proof_report_mode_pair(artifact_dir: Path, manifest: dict[str, Any], errors: list[str]) -> None:
    proof_path = artifact_dir / "proof_report.json"
    if not proof_path.is_file():
        return
    try:
        proof_payload = _load_json(proof_path)
    except Exception as exc:
        errors.append(f"proof_report.json unreadable: {exc}")
        return

    if proof_payload.get("bundle_contract") != manifest.get("bundle_contract"):
        errors.append("proof_report bundle_contract mismatch vs manifest")
    if proof_payload.get("export_proof") != manifest.get("export_proof"):
        errors.append("proof_report export_proof mismatch vs manifest")


def evaluate_gate_g5_manifest_valid(artifact_dir: Path, manifest: dict[str, Any], mode_errors: list[str]) -> dict[str, Any]:
    failures: list[str] = list(mode_errors)

    run_manifest_schema = _load_json(RUN_MANIFEST_SCHEMA_PATH)
    try:
        jsonschema.validate(instance=manifest, schema=run_manifest_schema)
    except jsonschema.ValidationError as exc:
        failures.append(f"schema violation: {exc.message}")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        failures.append("manifest artifacts is not an object")
    else:
        self_entry = artifacts.get("run_manifest.json")
        if self_entry != "self-unhashed":
            failures.append("run_manifest.json entry must be self-unhashed")

        _validate_proof_report_mode_pair(artifact_dir, manifest, failures)

        for name, expected in artifacts.items():
            if not isinstance(name, str) or not isinstance(expected, str):
                failures.append("artifact entries must be string:string")
                continue
            if name == "run_manifest.json":
                continue
            if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
                failures.append(f"{name} has malformed hash")
                continue
            path = artifact_dir / name
            if not path.is_file():
                failures.append(f"{name} missing on disk")
                continue
            if sha256_file(path) != expected:
                failures.append(f"{name} hash mismatch")

    return {"status": "PASS" if not failures else "FAIL", "details": "manifest schema + semantic mode + hash integrity", **({"errors": failures} if failures else {})}


def evaluate_gate_g6_determinism(artifact_dir: Path) -> dict[str, Any]:
    try:
        payload = _load_json(artifact_dir / "robustness_report.json")
        replay = payload.get("replay_check")
        if not isinstance(replay, dict):
            return {"status": "FAIL", "details": "missing replay_check"}
        identical = replay.get("identical") is True
        hashes = replay.get("hashes")
        if not isinstance(hashes, list) or not hashes:
            return {"status": "FAIL", "details": "missing replay hashes"}
        all_equal = all(isinstance(row, dict) and row.get("run_a") == row.get("run_b") for row in hashes)
        ok = identical and all_equal
        return {"status": "PASS" if ok else "FAIL", "details": "same-seed replay hash equality for required traces"}
    except Exception as exc:
        return {"status": "FAIL", "details": f"determinism gate unreadable: {exc}"}


def evaluate_gate_g7_avalanche(artifact_dir: Path) -> dict[str, Any]:
    try:
        payload = _load_json(artifact_dir / "avalanche_fit_report.json")
        validity = payload.get("validity")
        if not isinstance(validity, dict):
            return {"status": "FAIL", "details": "missing validity section"}
        reasons = validity.get("reasons", [])
        return {
            "status": "PASS" if validity.get("verdict") == "PASS" else "FAIL",
            "details": "avalanche fit evidence gate",
            **({"errors": reasons} if isinstance(reasons, list) and reasons else {}),
        }
    except Exception as exc:
        return {"status": "FAIL", "details": f"avalanche gate unreadable: {exc}"}


def evaluate_gate_g8_repro_envelope(artifact_dir: Path) -> dict[str, Any]:
    try:
        payload = _load_json(artifact_dir / "envelope_report.json")
        reasons = payload.get("failure_reasons", [])
        return {
            "status": "PASS" if payload.get("verdict") == "PASS" else "FAIL",
            "details": "10-seed canonical admissibility-band gate",
            **({"errors": reasons} if isinstance(reasons, list) and reasons else {}),
        }
    except Exception as exc:
        return {"status": "FAIL", "details": f"envelope gate unreadable: {exc}"}


def _gate_state(registry: dict[str, dict[str, Any]], gate_id: str) -> str:
    state = registry[gate_id].get("status")
    if state not in {"wired", "planned"}:
        raise ValueError(f"invalid gate status for {gate_id}: {state}")
    return str(state)


def _compute_verdict(gates: dict[str, dict[str, Any]], registry: dict[str, dict[str, Any]]) -> tuple[str, int, list[str]]:
    wired_failures = [gate_id for gate_id, payload in gates.items() if _gate_state(registry, gate_id) == "wired" and payload["status"] == "FAIL"]
    if wired_failures:
        return "FAIL", 2, [f"{gate_id} failed" for gate_id in wired_failures]

    unresolved = [gate_id for gate_id, payload in gates.items() if payload["status"] == "INCONCLUSIVE"]
    if unresolved:
        return "INCONCLUSIVE", 1, [f"{gate_id} unresolved" for gate_id in unresolved]
    return "PASS", 0, []


def _discover_seed(artifact_dir: Path) -> int:
    try:
        summary = _load_json(artifact_dir / "summary_metrics.json")
        value = summary.get("seed")
        if isinstance(value, int) and value >= 0:
            return value
    except (OSError, ValueError, json.JSONDecodeError):
        return 0
    return 0


def _fail_closed_report(artifact_dir: Path, reason: str) -> dict[str, Any]:
    gates = {gate_id: {"status": "FAIL", "details": f"fail-closed: {reason}"} for gate_id in EXPECTED_GATE_IDS}
    return {
        "schema_version": PROOF_SCHEMA_VERSION,
        "bundle_contract": CANONICAL_BASE_CONTRACT,
        "export_proof": False,
        "verdict": "FAIL",
        "verdict_code": 2,
        "timestamp_utc": DETERMINISTIC_TIMESTAMP_UTC,
        "seed": _discover_seed(artifact_dir),
        "gates": gates,
        "metrics": {},
        "recomputed_metrics": {},
        "summary_metrics_snapshot": {},
        "recompute_sources": {},
        "metric_consistency": {},
        "artifacts_verified": [],
        "failure_reasons": [f"fail-closed: {reason}"],
    }


def evaluate_all_gates(artifact_dir: str | Path) -> dict[str, Any]:
    artifact_root = Path(artifact_dir)
    try:
        loaded = load_artifacts(artifact_root)
        summary_metrics = loaded["summary"]
        manifest = loaded["manifest"]
        registry = loaded["registry"]
        recomputed = recompute_metrics_from_artifacts(artifact_root, summary_metrics, manifest)
        recomputed_metrics = recomputed.get("metrics", {})

        mode, mode_errors = mode_from_manifest(manifest)
        if mode is not None:
            required_artifacts = _required_artifacts_from_registry(registry, mode)
        else:
            export_hint = bool(manifest.get("export_proof"))
            fallback_mode = ManifestMode(
                bundle_contract="canonical-export-proof" if export_hint else "canonical-base",
                export_proof=export_hint,
                cmd=str(manifest.get("cmd", "")),
            )
            required_artifacts = _required_artifacts_from_registry(registry, fallback_mode)

        gates: dict[str, dict[str, Any]] = {
            "G1_active_spiking": evaluate_gate_g1_active_spiking(recomputed_metrics, registry["G1_active_spiking"]),
            "G2_rate_in_bounds": evaluate_gate_g2_rate_bounds(recomputed_metrics, registry["G2_rate_in_bounds"]),
            "G3_sigma_in_range": evaluate_gate_g3_sigma_range(recomputed_metrics, registry["G3_sigma_in_range"]),
            "G6_determinism_replay": evaluate_gate_g6_determinism(artifact_root),
            "G7_avalanche_evidence_sufficient": evaluate_gate_g7_avalanche(artifact_root),
            "G8_reproducibility_envelope": evaluate_gate_g8_repro_envelope(artifact_root),
            "G9_metrics_trace_consistency": evaluate_gate_g9_metric_consistency(summary_metrics, recomputed),
        }

        g4_result, artifacts_verified = evaluate_gate_g4_artifact_contract(artifact_root, manifest, mode_errors, required_artifacts)
        gates["G4_core_artifacts_complete"] = g4_result
        gates["G5_manifest_valid"] = evaluate_gate_g5_manifest_valid(artifact_root, manifest, mode_errors)

        ordered_gates = {gate_id: gates[gate_id] for gate_id in EXPECTED_GATE_IDS}
        verdict, verdict_code, failure_reasons = _compute_verdict(ordered_gates, registry)

        manifest_contract = manifest.get("bundle_contract")
        bundle_contract = manifest_contract if manifest_contract in {"canonical-base", "canonical-export-proof"} else CANONICAL_BASE_CONTRACT
        export_proof = manifest.get("export_proof") if isinstance(manifest.get("export_proof"), bool) else False

        report = {
            "schema_version": PROOF_SCHEMA_VERSION,
            "bundle_contract": bundle_contract,
            "export_proof": export_proof,
            "verdict": verdict,
            "verdict_code": verdict_code,
            "timestamp_utc": DETERMINISTIC_TIMESTAMP_UTC,
            "seed": int(summary_metrics.get("seed", manifest.get("seed", 0))),
            "gates": ordered_gates,
            "metrics": {
                "spike_events": int(recomputed_metrics.get("spike_events", 0)),
                "rate_mean_hz": float(recomputed_metrics.get("rate_mean_hz", 0.0)),
                "sigma_mean": float(recomputed_metrics.get("sigma_mean", 0.0)),
            },
            "recomputed_metrics": dict(recomputed_metrics),
            "summary_metrics_snapshot": {
                "spike_events": summary_metrics.get("spike_events"),
                "rate_mean_hz": summary_metrics.get("rate_mean_hz"),
                "sigma_mean": summary_metrics.get("sigma_mean"),
            },
            "recompute_sources": {
                "rate_mean_hz": recomputed.get("sources", {}).get("rate_mean_hz"),
                "sigma_mean": recomputed.get("sources", {}).get("sigma_mean"),
                "spike_events": recomputed.get("sources", {}).get("spike_events"),
                "spike_events_source": recomputed.get("spike_events_source"),
                **({"spike_events_reconstruction": recomputed["spike_events_reconstruction"]} if "spike_events_reconstruction" in recomputed else {}),
            },
            "metric_consistency": gates["G9_metrics_trace_consistency"].get("consistency", {}),
            "artifacts_verified": artifacts_verified,
            "failure_reasons": failure_reasons,
        }
    except Exception as exc:  # pragma: no cover
        report = _fail_closed_report(artifact_root, f"{exc.__class__.__name__}: {exc}")

    proof_schema = _load_json(PROOF_SCHEMA_PATH)
    jsonschema.validate(instance=report, schema=proof_schema)
    return report


def emit_proof_report(result: dict[str, Any], artifact_dir: str | Path) -> Path:
    artifact_root = Path(artifact_dir)
    proof_schema = _load_json(PROOF_SCHEMA_PATH)
    jsonschema.validate(instance=result, schema=proof_schema)
    report_path = artifact_root / "proof_report.json"
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report_path


def _update_manifest_proof_hash(artifact_dir: Path, proof_hash: str) -> None:
    manifest_path = artifact_dir / "run_manifest.json"
    manifest = _load_json(manifest_path)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("run_manifest artifacts must be object")
    artifacts["proof_report.json"] = proof_hash
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def evaluate_and_emit(artifact_dir: str | Path) -> EvaluationResult:
    root = Path(artifact_dir)
    manifest = _load_json(root / "run_manifest.json")
    export_requested = manifest.get("export_proof") is True

    if not export_requested:
        report = evaluate_all_gates(root)
        report_path = emit_proof_report(report, root)
        return EvaluationResult(report=report, report_path=report_path)

    provisional_report = evaluate_all_gates(root)
    provisional_path = emit_proof_report(provisional_report, root)
    _update_manifest_proof_hash(root, sha256_file(provisional_path))

    final_report = evaluate_all_gates(root)
    final_path = emit_proof_report(final_report, root)
    _update_manifest_proof_hash(root, sha256_file(final_path))

    consistency_report = evaluate_all_gates(root)
    if consistency_report != final_report:
        consistency_path = emit_proof_report(consistency_report, root)
        _update_manifest_proof_hash(root, sha256_file(consistency_path))
        terminal_report = evaluate_all_gates(root)
        if terminal_report != consistency_report:
            raise RuntimeError("proof/manifest finalization failed to stabilize")
        return EvaluationResult(report=terminal_report, report_path=consistency_path)

    return EvaluationResult(report=final_report, report_path=final_path)
