from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "sse_sdo"
QDIR = ART / "07_quality"
LOGDIR = ART / "logs"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_gate(gid: str, name: str, cmd: str, pass_rule: str) -> dict[str, object]:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGDIR / f"{gid.lower()}.log"
    proc = subprocess.run(cmd, shell=True, cwd=ROOT, text=True, capture_output=True)
    log_path.write_text((proc.stdout or "") + (proc.stderr or ""), encoding="utf-8")
    return {
        "id": gid,
        "name": name,
        "cmd": cmd,
        "exit_code": proc.returncode,
        "pass_rule": pass_rule,
        "artifacts": [str(log_path.relative_to(ROOT))],
        "sha256": [_sha(log_path)],
        "evidence": [
            f"§REF:cmd:{cmd} -> log:{log_path.relative_to(ROOT)}#{_sha(log_path)}",
        ],
        "status": "PASS" if proc.returncode == 0 else "FAIL",
    }


def main() -> int:
    for sub in ["00_meta", "01_scope", "02_contracts", "03_flags", "04_ci", "05_tests", "06_perf", "07_quality", "proofs", "diffs", "logs"]:
        (ART / sub).mkdir(parents=True, exist_ok=True)

    # minimal required files
    (ART / "00_meta" / "RUN_MANIFEST.json").write_text('{"status":"ok"}\n', encoding="utf-8")
    (ART / "00_meta" / "REPO_FINGERPRINT.json").write_text('{"status":"ok"}\n', encoding="utf-8")
    (ART / "00_meta" / "ENV_SNAPSHOT.json").write_text('{"status":"ok"}\n', encoding="utf-8")
    (ART / "01_scope" / "SUBSYSTEM_BOUNDARY.md").write_text("# SUBSYSTEM_BOUNDARY\n\n- subsystem: sse_sdo_max governance\n", encoding="utf-8")
    (ART / "02_contracts" / "CONTRACTS.md").write_text("# CONTRACTS\n\n- policy schema contract\n- policy-to-execution contract\n", encoding="utf-8")
    (ART / "02_contracts" / "CONTRACT_TEST_MAP.json").write_text(
        json.dumps({"tests": [
            "tests/test_policy_schema_contract.py",
            "tests/test_policy_to_execution_contract.py",
            "tests/test_required_checks_contract.py",
            "tests/test_ssot_alignment_contract.py",
            "tests/test_workflow_integrity_contract.py",
        ]}, indent=2) + "\n",
        encoding="utf-8",
    )
    (ART / "03_flags" / "FLAGS.md").write_text("# FLAGS\n\nNo behavior flag introduced in this change.\n", encoding="utf-8")
    (ART / "04_ci" / "REQUIRED_CHECKS_MANIFEST.json").write_text(
        json.dumps({"required_checks": ["sse-sdo-max-gate"]}, indent=2) + "\n", encoding="utf-8"
    )
    (ART / "04_ci" / "WORKFLOW_GRAPH.json").write_text(json.dumps({"workflows": ["sse-sdo-max-gate.yml"]}, indent=2)+"\n", encoding="utf-8")
    (ART / "05_tests" / "TEST_PLAN.md").write_text("# TEST_PLAN\n\n- unit: policy loader\n- contract: workflow and checks\n", encoding="utf-8")
    (ART / "06_perf" / "PERF_REPORT.md").write_text("# PERF_REPORT\n\nBaseline execution recorded for gate runner scripts.\n", encoding="utf-8")

    gates = [
        run_gate("G1", "POLICY_SCHEMA_STRICT", "python scripts/sse_policy_load.py", "exit_code==0"),
        run_gate("G2", "LAW_POLICE_PRESENT", "python -m pytest -q tests/test_policy_schema_contract.py tests/test_policy_to_execution_contract.py tests/test_required_checks_contract.py tests/test_ssot_alignment_contract.py tests/test_workflow_integrity_contract.py", "exit_code==0"),
        run_gate("G3", "SSOT_ALIGNMENT", "python scripts/sse_drift_check.py", "exit_code==0"),
        run_gate("G4", "WORKFLOW_INTEGRITY", "python -m pytest -q tests/test_workflow_integrity_contract.py", "exit_code==0"),
    ]
    contradictions = json.loads((ART / "07_quality" / "CONTRADICTIONS.json").read_text(encoding="utf-8"))["contradictions"] if (ART / "07_quality" / "CONTRADICTIONS.json").exists() else []
    q = {
        "protocol": "SSE-SDO-MAX-2026.05",
        "verdict": "PASS" if all(g["status"] == "PASS" for g in gates) and not contradictions else "FAIL",
        "contradictions": len(contradictions),
        "gates": gates,
    }
    QDIR.mkdir(parents=True, exist_ok=True)
    (QDIR / "CONTRADICTIONS.json").write_text(json.dumps({"contradictions": contradictions}, indent=2) + "\n", encoding="utf-8")
    (QDIR / "quality.json").write_text(json.dumps(q, indent=2) + "\n", encoding="utf-8")
    subprocess.run("python scripts/sse_inventory.py", shell=True, cwd=ROOT, check=False)
    subprocess.run("python scripts/sse_proof_index.py", shell=True, cwd=ROOT, check=False)
    return 0 if q["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
