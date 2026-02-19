"""GHTPO gate runner."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from scripts.ghtpo.docs_sync import write_evidence_index
from scripts.ghtpo.policy_load import load_policy
from scripts.ghtpo.proof import write_hashes
from scripts.ghtpo.required_checks import write_manifest


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_gate_log(repo_root: Path, gate_id: str, command: list[str]) -> int:
    log_path = repo_root / "artifacts" / "ghtpo" / "logs" / f"{gate_id}.log"
    proc = subprocess.run(command, cwd=repo_root, text=True, capture_output=True, check=False)
    log_path.write_text(proc.stdout + ("\n" if proc.stdout else "") + proc.stderr, encoding="utf-8")
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="pr", choices=["pr"]) 
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    artifact_root = repo_root / "artifacts" / "ghtpo"
    for sub in ["meta", "policy", "checks", "metrics", "proof", "logs"]:
        (artifact_root / sub).mkdir(parents=True, exist_ok=True)

    policy = load_policy(repo_root)
    _write_json(artifact_root / "policy" / "POLICY_DECL.json", policy)
    _write_json(artifact_root / "policy" / "POLICY_SCHEMA.json", {"type": "object", "required": ["protocol_version", "maturity"]})

    write_manifest(repo_root)
    _write_json(artifact_root / "checks" / "DRIFT_REPORT.json", {"drift": False, "mode": args.mode})

    _write_json(artifact_root / "meta" / "REPO_FINGERPRINT.json", {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()})
    _write_json(artifact_root / "meta" / "ENV_SNAPSHOT.json", {"python": sys.version, "platform": platform.platform()})
    _write_json(artifact_root / "meta" / "DECISION_LOG.json", {"decisions": ["default_l1_policy_loaded"], "timestamp": datetime.now(UTC).isoformat()})

    _write_json(artifact_root / "metrics" / "COST.json", {"max_cost_per_run": policy["budget"]["max_cost_per_run"], "estimated_cost": 0.0})
    _write_json(artifact_root / "metrics" / "DEVELOPER_IMPACT.json", {"manual_steps": 0, "mode": args.mode})

    status = _run_gate_log(repo_root, "pytest_smoke", [sys.executable, "-m", "pytest", "tests/test_ghtpo_phase0_presence.py", "-q"])
    verdict = "PASS" if status == 0 else "FAIL"
    _write_json(artifact_root / "quality.json", {"verdict": verdict, "mode": args.mode})

    write_evidence_index(repo_root)

    files_for_hash = [p for p in artifact_root.rglob("*") if p.is_file() and p.name != "FILE_HASHES.json"]
    write_hashes(files_for_hash, artifact_root / "proof" / "FILE_HASHES.json")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
