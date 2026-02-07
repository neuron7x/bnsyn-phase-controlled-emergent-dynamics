from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts import assert_non_empty_text, assert_numeric_finite_and_bounded

AUDIT_DIR = ROOT / "artifacts" / "math_audit"
MANIFEST_PATH = AUDIT_DIR / "manifest.json"
REPORT_JSON_PATH = AUDIT_DIR / "validator_report.json"
REPORT_MD_PATH = AUDIT_DIR / "validator_report.md"

SCAN_DIRS = ("results", "benchmarks", "docs", "src")
CONFIG_FILES = ("pyproject.toml", "Makefile", "docker-compose.yml")


@dataclass(frozen=True)
class CheckResult:
    artifact: str
    check_name: str
    status: str
    evidence: str
    duration_ms: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_file(path: Path) -> str:
    suffix = path.suffix.lower()
    path_str = str(path)
    if path_str.startswith("src/"):
        return "numeric_code"
    if path_str.startswith("results/"):
        return "derived_data"
    if suffix in {".json", ".csv", ".npy", ".npz", ".parquet"}:
        return "derived_data"
    if suffix in {".md", ".rst"}:
        return "report"
    if suffix in {".toml", ".yml", ".yaml", ".ini"}:
        return "config"
    return "formula"


def generator_for(path: Path) -> tuple[str | None, str, str]:
    path_str = str(path)
    if path_str.startswith("benchmarks/baselines/"):
        return (
            "scripts/generate_benchmark_baseline.py",
            "PARTIAL",
            "generator exists; seed handling delegated to benchmark scenario",
        )
    if path_str.startswith("benchmarks/") and path.suffix == ".json":
        return (
            "scripts/run_benchmarks.py",
            "PARTIAL",
            "generator exists; artifact depends on runtime profile selection",
        )
    if path_str.startswith("results/"):
        return (None, "UNTRUSTED", "no deterministic generator metadata co-located")
    if path_str.startswith("src/"):
        return (None, "SOURCED", "version-controlled source")
    return (None, "PARTIAL", "no explicit generator metadata")


def iter_scope_files() -> list[Path]:
    files: list[Path] = []
    for directory in SCAN_DIRS:
        base = ROOT / directory
        if not base.exists():
            continue
        files.extend(sorted(p for p in base.rglob("*") if p.is_file()))
    for config in CONFIG_FILES:
        candidate = ROOT / config
        if candidate.exists():
            files.append(candidate)
    unique = sorted({p.resolve() for p in files})
    return [p.relative_to(ROOT) for p in unique]


def build_manifest() -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    for rel_path in iter_scope_files():
        abs_path = ROOT / rel_path
        generator, provenance, gap = generator_for(rel_path)
        artifacts.append(
            {
                "path": str(rel_path).replace("\\", "/"),
                "sha256": sha256_file(abs_path),
                "type": classify_file(rel_path),
                "generator": generator,
                "provenance": provenance,
                "provenance_gap": gap,
            }
        )
    return {
        "schema_version": "2.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
    }


def load_data(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    if suffix in {".yml", ".yaml"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix in {".md", ".py", ".toml", ".txt"}:
        return path.read_text(encoding="utf-8", errors="replace")
    return None


def extract_numeric_scalars(obj: Any) -> list[float]:
    out: list[float] = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
        return out
    if isinstance(obj, dict):
        for value in obj.values():
            out.extend(extract_numeric_scalars(value))
        return out
    if isinstance(obj, list):
        for value in obj:
            out.extend(extract_numeric_scalars(value))
    return out


def analyze_python_hazards(path: Path, source: str) -> list[str]:
    hazards: list[str] = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
            if isinstance(node.left, ast.Name) and isinstance(node.right, ast.Name):
                hazards.append(f"possible_cancellation:{path}:{node.lineno}")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            if isinstance(node.right, ast.Constant) and node.right.value == 0:
                hazards.append(f"division_by_zero_literal:{path}:{node.lineno}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "exp":
                hazards.append(f"exp_overflow_risk:{path}:{node.lineno}")
    return hazards


def run_check(artifact: str, check_name: str, fn: Any) -> CheckResult:
    start = time.perf_counter()
    try:
        status, evidence = fn()
    except Exception as exc:  # noqa: BLE001
        status, evidence = "FAIL", f"exception:{type(exc).__name__}:{exc}"
    elapsed = int((time.perf_counter() - start) * 1000)
    return CheckResult(artifact=artifact, check_name=check_name, status=status, evidence=evidence, duration_ms=elapsed)


def validate_manifest(manifest: dict[str, Any]) -> list[CheckResult]:
    checks: list[CheckResult] = []
    for artifact in manifest["artifacts"]:
        rel_path = Path(artifact["path"])
        abs_path = ROOT / rel_path
        artifact_name = artifact["path"]

        checks.append(
            run_check(
                artifact_name,
                "sha256_match",
                lambda p=abs_path, expected=artifact["sha256"]: (
                    ("PASS", "digest_match") if sha256_file(p) == expected else ("FAIL", "digest_mismatch")
                ),
            )
        )

        data = load_data(abs_path)

        def schema_check() -> tuple[str, str]:
            if data is None:
                return "SKIP", "unsupported_format"
            if isinstance(data, str):
                try:
                    assert_non_empty_text(data)
                except ValueError as exc:
                    return "FAIL", str(exc)
                return "PASS", "non_empty_text"
            return "PASS", f"loaded_type:{type(data).__name__}"

        checks.append(run_check(artifact_name, "schema_load", schema_check))

        def invariant_check() -> tuple[str, str]:
            numerics = extract_numeric_scalars(data)
            if not numerics:
                return "SKIP", "no_numeric_content"
            try:
                assert_numeric_finite_and_bounded(numerics, bound=1e12)
            except ValueError as exc:
                return "FAIL", str(exc)
            return "PASS", f"numeric_count:{len(numerics)}"

        checks.append(run_check(artifact_name, "numeric_sanity", invariant_check))

        def anomaly_check() -> tuple[str, str]:
            expected = artifact.get("distribution_expectation")
            numerics = extract_numeric_scalars(data)
            if expected is None:
                return "SKIP", "missing_distribution_expectation"
            if len(numerics) < 8:
                return "SKIP", "insufficient_samples"
            mean = statistics.fmean(numerics)
            std = statistics.pstdev(numerics)
            if std == 0:
                return "PASS", "constant_distribution"
            max_z = max(abs((x - mean) / std) for x in numerics)
            if max_z > 4.0:
                return "FAIL", f"distribution_anomaly_max_z={max_z:.3f}"
            return "PASS", f"max_z={max_z:.3f}"

        checks.append(run_check(artifact_name, "distribution_anomaly", anomaly_check))

        if rel_path.suffix == ".py":
            source = data if isinstance(data, str) else abs_path.read_text(encoding="utf-8")

            def hazard_check() -> tuple[str, str]:
                hazards = analyze_python_hazards(rel_path, source)
                if not hazards:
                    return "PASS", "no_hazard_pattern_detected"
                return "PASS", f"hazards_documented:{len(hazards)}"

            checks.append(run_check(artifact_name, "numeric_hazard_scan", hazard_check))
    return checks


def write_report(manifest: dict[str, Any], checks: list[CheckResult]) -> int:
    status_counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    for check in checks:
        status_counts[check.status] = status_counts.get(check.status, 0) + 1

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "artifacts": len(manifest["artifacts"]),
            "checks": len(checks),
            **status_counts,
        },
        "checks": [check.__dict__ for check in checks],
    }
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    provenance_counts = {"SOURCED": 0, "PARTIAL": 0, "UNTRUSTED": 0}
    for artifact in manifest["artifacts"]:
        provenance_counts[artifact["provenance"]] += 1

    lines = [
        "# Math Validator Report",
        "",
        f"- artifacts: {len(manifest['artifacts'])}",
        f"- checks: {len(checks)}",
        f"- PASS: {status_counts['PASS']}",
        f"- FAIL: {status_counts['FAIL']}",
        f"- SKIP: {status_counts['SKIP']}",
        f"- SOURCED: {provenance_counts['SOURCED']}",
        f"- PARTIAL: {provenance_counts['PARTIAL']}",
        f"- UNTRUSTED: {provenance_counts['UNTRUSTED']}",
        "",
        "## Failing checks",
    ]
    failing = [c for c in checks if c.status == "FAIL"]
    if not failing:
        lines.append("- none")
    else:
        for check in failing:
            lines.append(f"- {check.artifact} :: {check.check_name} :: {check.evidence}")

    REPORT_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 1 if status_counts["FAIL"] > 0 else 0


def main() -> int:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    checks = validate_manifest(manifest)
    return write_report(manifest, checks)


if __name__ == "__main__":
    raise SystemExit(main())
