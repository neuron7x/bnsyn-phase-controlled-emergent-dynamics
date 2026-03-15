#!/usr/bin/env python3
"""Generate a release readiness report for BN-Syn."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tomllib

from tools.entropy_gate.compute_metrics import compute_metrics, flatten


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    details: str
    blocking: bool


def check_files_exist(name: str, paths: list[Path], blocking: bool = True) -> CheckResult:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        return CheckResult(
            name=name,
            status="fail",
            details=f"Missing files: {', '.join(missing)}",
            blocking=blocking,
        )
    return CheckResult(name=name, status="pass", details="All files present", blocking=blocking)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_mutation_baseline(path: Path) -> CheckResult:
    if not path.exists():
        return CheckResult(
            name="Mutation baseline",
            status="fail",
            details=f"Missing {path}",
            blocking=True,
        )
    data = load_json(path)
    status = data.get("status")
    metrics = data.get("metrics", {})
    total_mutants = metrics.get("total_mutants")
    killed_mutants = metrics.get("killed_mutants")
    if status != "active" or not isinstance(total_mutants, int) or total_mutants <= 0:
        detail_parts = [
            f"status={status!r}",
            f"total_mutants={total_mutants!r}",
        ]
        return CheckResult(
            name="Mutation baseline",
            status="fail",
            details="Baseline not active or missing factual counts ("
            + ", ".join(detail_parts)
            + ")",
            blocking=True,
        )
    if not isinstance(killed_mutants, int) or killed_mutants <= 0:
        return CheckResult(
            name="Mutation baseline",
            status="fail",
            details=(
                "Baseline is trivial: metrics.killed_mutants must be > 0 "
                f"(killed_mutants={killed_mutants!r})"
            ),
            blocking=True,
        )
    return CheckResult(
        name="Mutation baseline",
        status="pass",
        details=(
            f"status={status}, total_mutants={total_mutants}, "
            f"killed_mutants={killed_mutants}"
        ),
        blocking=True,
    )


def check_entropy_gate(repo_root: Path) -> CheckResult:
    policy_path = repo_root / "entropy" / "policy.json"
    baseline_path = repo_root / "entropy" / "baseline.json"
    if not policy_path.exists() or not baseline_path.exists():
        missing: list[str] = []
        if not policy_path.exists():
            missing.append(str(policy_path))
        if not baseline_path.exists():
            missing.append(str(baseline_path))
        return CheckResult(
            name="Entropy gate",
            status="fail",
            details=f"Missing entropy gate files: {', '.join(missing)}",
            blocking=True,
        )

    policy = load_json(policy_path)
    baseline = load_json(baseline_path)
    comparators = policy.get("comparators", {})
    if not isinstance(comparators, dict) or not comparators:
        return CheckResult(
            name="Entropy gate",
            status="fail",
            details="policy.json comparators missing/empty",
            blocking=True,
        )

    current = compute_metrics(repo_root)
    baseline_flat = flatten(baseline)
    current_flat = flatten(current)

    failures: list[str] = []
    for key, comparator in sorted(comparators.items()):
        if key not in baseline_flat:
            failures.append(f"{key}: baseline missing key")
            continue
        if key not in current_flat:
            failures.append(f"{key}: current missing key")
            continue

        baseline_value = baseline_flat[key]
        current_value = current_flat[key]
        if comparator == "lte":
            if not (current_value <= baseline_value):
                failures.append(
                    f"{key}: regression (current={current_value} > baseline={baseline_value})"
                )
        elif comparator == "gte":
            if not (current_value >= baseline_value):
                failures.append(
                    f"{key}: regression (current={current_value} < baseline={baseline_value})"
                )
        elif comparator == "eq":
            if not (current_value == baseline_value):
                failures.append(
                    f"{key}: changed (current={current_value} != baseline={baseline_value})"
                )
        else:
            failures.append(f"{key}: unknown comparator '{comparator}'")

    if failures:
        return CheckResult(
            name="Entropy gate",
            status="fail",
            details="; ".join(failures[:3]),
            blocking=True,
        )

    return CheckResult(
        name="Entropy gate",
        status="pass",
        details="No comparator regressions vs entropy baseline",
        blocking=True,
    )


def check_pyproject_version(path: Path) -> tuple[CheckResult, str | None]:
    if not path.exists():
        return (
            CheckResult(
                name="Project version",
                status="fail",
                details=f"Missing {path}",
                blocking=True,
            ),
            None,
        )
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    version = data.get("project", {}).get("version")
    if not version:
        return (
            CheckResult(
                name="Project version",
                status="fail",
                details="Missing [project].version in pyproject.toml",
                blocking=True,
            ),
            None,
        )
    return (
        CheckResult(
            name="Project version",
            status="pass",
            details=f"version={version}",
            blocking=True,
        ),
        str(version),
    )


def build_report(repo_root: Path) -> dict[str, Any]:
    required_repo_files = [
        repo_root / "README.md",
        repo_root / "LICENSE",
        repo_root / "SECURITY.md",
        repo_root / "CITATION.cff",
        repo_root / "requirements-lock.txt",
    ]
    governance_files = [
        repo_root / "VERIFICATION_REPORT.md",
        repo_root / "GOVERNANCE_VERIFICATION_REPORT.md",
        repo_root / "HARDENING_SUMMARY.md",
        repo_root / "README_CLAIMS_GATE.md",
    ]
    quality_docs = [
        repo_root / "docs" / "QUALITY_INFRASTRUCTURE.md",
        repo_root / "docs" / "CI_GATES.md",
        repo_root / "docs" / "TESTING_MUTATION.md",
    ]
    quality_scripts = [
        repo_root / "scripts" / "lint_ci_truthfulness.py",
        repo_root / "scripts" / "verify_formal_constants.py",
        repo_root / "scripts" / "generate_mutation_baseline.py",
        repo_root / "scripts" / "check_mutation_score.py",
    ]

    checks: list[CheckResult] = []
    checks.append(check_files_exist("Core repository files", required_repo_files, blocking=True))
    checks.append(check_files_exist("Governance evidence", governance_files, blocking=True))
    checks.append(check_files_exist("Quality documentation", quality_docs, blocking=True))
    checks.append(check_files_exist("Quality scripts", quality_scripts, blocking=True))

    version_check, version = check_pyproject_version(repo_root / "pyproject.toml")
    checks.append(version_check)
    checks.append(check_mutation_baseline(repo_root / "quality" / "mutation_baseline.json"))
    checks.append(check_entropy_gate(repo_root))

    blocking_failures = [
        check.name for check in checks if check.blocking and check.status != "pass"
    ]
    release_ready = len(blocking_failures) == 0

    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "timestamp": timestamp,
        "release_ready": release_ready,
        "version": version,
        "blocking_failures": blocking_failures,
        "checks": [
            {
                "name": check.name,
                "status": check.status,
                "blocking": check.blocking,
                "details": check.details,
            }
            for check in checks
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    status = "READY" if report["release_ready"] else "BLOCKED"
    lines = [
        "# Release Readiness Report",
        "",
        f"- Timestamp: {report['timestamp']}",
        f"- Version: {report.get('version') or 'unknown'}",
        f"- Overall status: **{status}**",
        "",
        "| Check | Status | Blocking | Details |",
        "| --- | --- | --- | --- |",
    ]
    for check in report["checks"]:
        lines.append(
            f"| {check['name']} | {check['status']} | {check['blocking']} | {check['details']} |"
        )
    if report["blocking_failures"]:
        lines.extend(
            [
                "",
                "## Blocking Failures",
                "",
            ]
        )
        for failure in report["blocking_failures"]:
            lines.append(f"- {failure}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate release readiness report.")
    parser.add_argument(
        "--json-out",
        default="artifacts/release_readiness.json",
        help="Path to write JSON report",
    )
    parser.add_argument(
        "--md-out",
        default="artifacts/release_readiness.md",
        help="Path to write Markdown report",
    )
    parser.add_argument(
        "--advisory",
        action="store_true",
        help="Exit 0 even if blocking checks fail",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    report = build_report(repo_root)

    json_path = repo_root / args.json_out
    md_path = repo_root / args.md_out
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report) + "\n", encoding="utf-8")

    print(f"Release readiness: {'READY' if report['release_ready'] else 'BLOCKED'}")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")

    if report["release_ready"] or args.advisory:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
