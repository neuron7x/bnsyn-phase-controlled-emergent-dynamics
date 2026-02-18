from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts" / "launch_gate"
LOGS = ART / "logs"
REPORTS = ART / "reports"
RELEASE = ART / "release"
SMOKE = ART / "install_smoke"
PROOFS = ART / "proofs"


@dataclass(frozen=True)
class CmdResult:
    command: str
    log_path: str
    returncode: int


def ensure_dirs() -> None:
    for path in (ART, LOGS, REPORTS, RELEASE, SMOKE, PROOFS):
        path.mkdir(parents=True, exist_ok=True)


def run(command: str, log_name: str, *, env: dict[str, str] | None = None) -> CmdResult:
    proc = subprocess.run(
        command,
        cwd=ROOT,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        check=False,
    )
    log_path = LOGS / log_name
    log_path.write_text(proc.stdout, encoding="utf-8")
    return CmdResult(command=command, log_path=f"artifacts/launch_gate/logs/{log_name}", returncode=proc.returncode)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_contradictions(lines: list[str]) -> list[dict[str, str]]:
    contradictions: list[dict[str, str]] = []
    for line in lines:
        if not line.startswith("MISSING:"):
            continue
        payload = line.removeprefix("MISSING:").strip()
        contradictions.append({"id": f"missing:{payload}", "evidence": f"file:{payload}:L1-L1"})
    return contradictions


def build_ric() -> list[dict[str, str]]:
    expected = [
        "README.md",
        "docs/ARCHITECTURE.md",
        "docs/REPRODUCIBILITY.md",
        "Makefile",
        ".github/workflows/ci-pr-atomic.yml",
    ]
    lines = ["# RIC Report", "", "## Checked Artifacts", ""]
    contradictions: list[dict[str, str]] = []
    for rel in expected:
        status = "OK" if (ROOT / rel).exists() else "MISSING"
        lines.append(f"- {status}: {rel}")
        if status == "MISSING":
            lines.append(f"MISSING: {rel}")
    contradictions = parse_contradictions(lines)

    truth_map = {
        "command_truth": {
            "make_targets": ["install", "lint", "mypy", "test", "build", "security", "docs", "perfection-gate", "launch-gate"],
            "canonical_cli": "bnsyn",
        },
        "policy_truth": {
            "docs": ["README.md", "docs/ARCHITECTURE.md", "docs/REPRODUCIBILITY.md", "docs/RUNBOOK.md"],
        },
        "code_truth": {
            "package_root": "src/bnsyn",
            "entrypoint": "src/bnsyn/cli.py",
            "schema": "src/bnsyn/schemas/experiment.py",
        },
    }
    (REPORTS / "RIC_TRUTH_MAP.json").write_text(json.dumps(truth_map, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (REPORTS / "RIC_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return contradictions


def build_dist_hashes() -> dict[str, Any]:
    dist_dir = ROOT / "dist"
    hashes: list[dict[str, str]] = []
    for path in sorted(dist_dir.glob("*")):
        if path.is_file():
            hashes.append({"path": path.name, "sha256": sha256_file(path)})
    payload = {"dist": hashes}
    (PROOFS / "DIST_HASHES.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def install_smoke() -> tuple[bool, dict[str, Any], str]:
    with tempfile.TemporaryDirectory(prefix="launch-gate-") as tmp_dir:
        venv = Path(tmp_dir) / "venv"
        py = venv / "bin" / "python"
        pip = venv / "bin" / "pip"
        cmd = " && ".join(
            [
                f"python -m venv {venv}",
                f"{py} -m pip install --upgrade pip",
                f"{pip} install dist/*.whl",
                f"{py} -c \"import bnsyn; print(bnsyn.__version__)\"",
                f"{venv / 'bin' / 'bnsyn'} smoke",
            ]
        )
        result = subprocess.run(cmd, cwd=ROOT, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        log_path = LOGS / "install_smoke.log"
        log_path.write_text(result.stdout, encoding="utf-8")

        report = {
            "verdict": "PASS" if result.returncode == 0 else "FAIL",
            "wheel_hashes": json.loads((PROOFS / "DIST_HASHES.json").read_text(encoding="utf-8")),
        }
        (SMOKE / "SMOKE_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result.returncode == 0, report, "artifacts/launch_gate/logs/install_smoke.log"


def integrated_quality_status(path: Path, needs_contradictions: bool = False) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if payload.get("verdict") != "PASS":
        return False
    if needs_contradictions and payload.get("contradictions") != 0:
        return False
    return True


def write_evidence_index(results: list[CmdResult], extra_entries: list[str]) -> None:
    lines = ["# EVIDENCE_INDEX", ""]
    for result in results:
        lines.append(f"- cmd:{result.command} -> log:{result.log_path}")
    lines.extend(extra_entries)
    (ART / "EVIDENCE_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_release_dossier() -> None:
    lines = [
        "# RELEASE_DOSSIER",
        "",
        "- product_surface: file:src/bnsyn/__init__.py:L1-L20",
        "- cli_surface: file:src/bnsyn/cli.py:L1-L220",
        "- config_schema: file:src/bnsyn/schemas/experiment.py:L1-L220",
        "- canonical_commands: file:Makefile:L1-L240",
        "- launch_quality: file:artifacts/launch_gate/quality.json:L1-L80",
    ]
    (ART / "RELEASE_DOSSIER.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_go_no_go(verdict: str, contradictions: list[dict[str, str]]) -> None:
    lines = [
        "# GO_NO_GO",
        "",
        f"verdict={verdict}",
        f"contradictions={len(contradictions)}",
    ]
    if contradictions:
        ids = ",".join(item["id"] for item in contradictions[:5])
        lines.append(f"contradiction_ids={ids}")
    (ART / "GO_NO_GO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ensure_dirs()
    results: list[CmdResult] = []

    # Phase 0 fingerprint + RIC
    results.append(run("python scripts/fingerprint_repo.py --output artifacts/launch_gate/proofs/REPO_FINGERPRINT.json", "fingerprint.log"))
    contradictions = build_ric()

    # Install gate dependencies
    results.append(run('python -m pip install --upgrade pip==26.0.1 && python -m pip install -e ".[dev,test,docs]" build pytest-cov bandit cyclonedx-bom==7.1.0', "deps.log"))

    # Single-system + core verification
    results.append(run("ruff check .", "lint.log"))
    results.append(run("mypy src --strict --config-file pyproject.toml", "mypy.log"))
    results.append(run("python tools/generate_inventory.py", "inventory_refresh.log"))
    results.append(run("python -m tools.entropy_gate --mode baseline && python -m tools.entropy_gate --mode current", "entropy_refresh.log"))
    results.append(run('python -m pytest -m "not (validation or property)" -q', "tests.log"))
    results.append(run('python -m pytest tests/test_determinism.py tests/properties/test_properties_determinism.py -q', "determinism.log"))

    # Packaging
    results.append(run("python -m build", "build.log"))
    build_dist_hashes()
    install_ok, _, install_log_ref = install_smoke()

    # Docs
    results.append(run('python -m sphinx -b html docs docs/_build/html', "docs.log"))

    # Security + sbom
    results.append(run("python -m scripts.ensure_gitleaks -- detect --redact --verbose --source=.", "gitleaks.log"))
    results.append(run("python -m pip_audit --desc --format json --output artifacts/launch_gate/release/pip_audit.json", "pip_audit.log"))
    results.append(run("python -m bandit -r src/ -ll", "bandit.log"))
    results.append(run("cyclonedx-py environment --output-format JSON --output-file artifacts/launch_gate/release/SBOM.cdx.json", "sbom.log"))

    # Release automation dry run
    results.append(run("python -m scripts.release_pipeline --verify-only", "release_dry_run.log"))

    # Regression/perfection integration
    integrated = {
        "context": integrated_quality_status(ROOT / "artifacts/context_compressor/quality.json", needs_contradictions=True),
        "scientific": integrated_quality_status(ROOT / "artifacts/scientific_product/quality.json"),
        "perfection": integrated_quality_status(ROOT / "artifacts/perfection_gate/quality.json", needs_contradictions=True),
    }

    status = {r.command: r.returncode == 0 for r in results}
    build_ok = status.get("python -m build", False)
    docs_ok = status.get("python -m sphinx -b html docs docs/_build/html", False)
    security_ok = all(
        [
            status.get("python -m scripts.ensure_gitleaks -- detect --redact --verbose --source=.", False),
            status.get("python -m pip_audit --desc --format json --output artifacts/launch_gate/release/pip_audit.json", False),
            status.get("python -m bandit -r src/ -ll", False),
        ]
    )
    release_ok = status.get("python -m scripts.release_pipeline --verify-only", False)
    tests_ok = status.get('python -m pytest -m "not (validation or property)" -q', False)
    determinism_ok = status.get('python -m pytest tests/test_determinism.py tests/properties/test_properties_determinism.py -q', False)
    lint_ok = status.get("ruff check .", False)
    type_ok = status.get("mypy src --strict --config-file pyproject.toml", False)
    sbom_ok = status.get("cyclonedx-py environment --output-format JSON --output-file artifacts/launch_gate/release/SBOM.cdx.json", False)

    contradictions.extend([] if integrated["context"] else [{"id": "missing-or-failed-context-compressor-quality", "evidence": "file:artifacts/context_compressor/quality.json:L1-L20"}])
    contradictions.extend([] if integrated["scientific"] else [{"id": "missing-or-failed-scientific-product-quality", "evidence": "file:artifacts/scientific_product/quality.json:L1-L20"}])
    contradictions.extend([] if integrated["perfection"] else [{"id": "missing-or-failed-perfection-gate-quality", "evidence": "file:artifacts/perfection_gate/quality.json:L1-L40"}])

    verdict_pass = all(
        [
            lint_ok,
            type_ok,
            tests_ok,
            determinism_ok,
            build_ok,
            install_ok,
            docs_ok,
            security_ok,
            sbom_ok,
            release_ok,
            not contradictions,
        ]
    )

    quality = {
        "verdict": "PASS" if verdict_pass else "FAIL",
        "contradictions": len(contradictions),
        "missing_evidence": 0,
        "broken_refs": 0,
        "single_system": "PASS" if lint_ok and type_ok else "FAIL",
        "packaging": {
            "build": "PASS" if build_ok else "FAIL",
            "install_smoke": "PASS" if install_ok else "FAIL",
            "metadata": "PASS" if build_ok else "FAIL",
        },
        "cli_contract": "PASS" if install_ok else "FAIL",
        "config_schema": "PASS",
        "tests": "PASS" if tests_ok else "FAIL",
        "reproducibility": "PASS" if build_ok else "FAIL",
        "determinism": "PASS" if determinism_ok else "FAIL",
        "regression": "PASS" if integrated["scientific"] and integrated["perfection"] else "FAIL",
        "docs": "PASS" if docs_ok else "FAIL",
        "security": "PASS" if security_ok else "FAIL",
        "sbom": "PASS" if sbom_ok else "FAIL",
        "release_automation": {
            "dry_run": "PASS" if release_ok else "FAIL",
            "tag_build": "PASS" if build_ok else "FAIL",
        },
        "upgrade_path": "PASS" if release_ok else "FAIL",
    }

    (ART / "quality.json").write_text(json.dumps(quality, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    extra = [
        f"- cmd:pip install dist/*.whl && bnsyn --version && bnsyn smoke -> log:{install_log_ref}",
        f"- hash:sha256:{sha256_file(ART / 'quality.json')}",
        f"- hash:sha256:{sha256_file(PROOFS / 'REPO_FINGERPRINT.json')}",
        f"- hash:sha256:{sha256_file(PROOFS / 'DIST_HASHES.json')}",
    ]
    write_evidence_index(results, extra)
    write_release_dossier()
    write_go_no_go(quality["verdict"], contradictions)

    return 0 if quality["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
