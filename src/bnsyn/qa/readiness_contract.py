"""Execution-backed repository readiness contract for BN-Syn.

This module is the single source of truth for release readiness state.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tomllib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, Sequence

from tools.entropy_gate.compute_metrics import compute_metrics, flatten

TRUTH_MODEL_VERSION = "1.0.0"
_MAX_OUTPUT_CHARS = 4000
_STATUS_DOC_TOKENS = ("blocked", "advisory", "ready")


class ReadinessStatus(StrEnum):
    """Machine-readable readiness states."""

    BLOCKED = "blocked"
    ADVISORY = "advisory"
    READY = "ready"


@dataclass(frozen=True)
class ReadinessCheck:
    """Serializable readiness check result."""

    name: str
    kind: str
    status: str
    blocking: bool
    details: str
    command: str | None = None
    executed_command: list[str] | None = None
    exit_code: int | None = None
    duration_seconds: float | None = None
    stdout_excerpt: str | None = None
    stderr_excerpt: str | None = None
    evidence: str | None = None


@dataclass(frozen=True)
class ReadinessSubsystem:
    """Readiness subsystem containing related checks."""

    key: str
    label: str
    status: ReadinessStatus
    checks: tuple[ReadinessCheck, ...]


@dataclass(frozen=True)
class ReadinessState:
    """Aggregated readiness state for the repository."""

    truth_model_version: str
    timestamp: str
    version: str | None
    status: ReadinessStatus
    release_ready: bool
    execution_backed_pass_count: int
    blocking_failures: tuple[str, ...]
    advisory_findings: tuple[str, ...]
    subsystems: tuple[ReadinessSubsystem, ...]

    def to_report(self) -> dict[str, Any]:
        """Convert state to a JSON-serializable report."""
        return {
            "truth_model_version": self.truth_model_version,
            "timestamp": self.timestamp,
            "version": self.version,
            "state": self.status.value,
            "release_ready": self.release_ready,
            "execution_backed_pass_count": self.execution_backed_pass_count,
            "blocking_failures": list(self.blocking_failures),
            "advisory_findings": list(self.advisory_findings),
            "subsystems": [
                {
                    "key": subsystem.key,
                    "label": subsystem.label,
                    "status": subsystem.status.value,
                    "checks": [asdict(check) for check in subsystem.checks],
                }
                for subsystem in self.subsystems
            ],
        }

    @classmethod
    def evaluate(
        cls,
        repo_root: Path,
        *,
        command_runner: "CommandRunner | None" = None,
        proof_output_dir: Path | None = None,
    ) -> "ReadinessState":
        """Compute the repository readiness state."""
        runner = command_runner or SubprocessCommandRunner()
        output_dir = proof_output_dir or repo_root / "artifacts" / "release_readiness_bundle"

        static_quality = _evaluate_static_quality(repo_root, runner)
        runtime_proof = _evaluate_runtime_proof_path(repo_root, output_dir, runner)
        bundle_validation = _evaluate_bundle_validation(repo_root, output_dir, runner)
        governance = _evaluate_governance_consistency(repo_root)

        subsystems = (static_quality, runtime_proof, bundle_validation, governance)
        blocking_failures = tuple(
            f"{subsystem.label}: {check.name}"
            for subsystem in subsystems
            for check in subsystem.checks
            if check.blocking and check.status != "pass"
        )
        advisory_findings = tuple(
            f"{subsystem.label}: {check.name}"
            for subsystem in subsystems
            for check in subsystem.checks
            if not check.blocking and check.status != "pass"
        )
        execution_backed_pass_count = sum(
            1
            for subsystem in subsystems
            for check in subsystem.checks
            if check.kind == "command" and check.status == "pass"
        )

        if blocking_failures or execution_backed_pass_count == 0:
            status = ReadinessStatus.BLOCKED
        elif advisory_findings:
            status = ReadinessStatus.ADVISORY
        else:
            status = ReadinessStatus.READY

        return cls(
            truth_model_version=TRUTH_MODEL_VERSION,
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            version=_read_project_version(repo_root / "pyproject.toml"),
            status=status,
            release_ready=status is ReadinessStatus.READY,
            execution_backed_pass_count=execution_backed_pass_count,
            blocking_failures=blocking_failures,
            advisory_findings=advisory_findings,
            subsystems=subsystems,
        )


@dataclass(frozen=True)
class CommandSpec:
    """Command execution specification."""

    name: str
    command: str
    argv: tuple[str, ...]
    blocking: bool = True
    cwd: Path | None = None


@dataclass(frozen=True)
class CommandOutcome:
    """Captured command execution outcome."""

    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float


class CommandRunner(Protocol):
    """Protocol for executing readiness commands."""

    def run(self, spec: CommandSpec, repo_root: Path) -> CommandOutcome:
        """Execute the given command spec."""


class SubprocessCommandRunner:
    """Default subprocess-backed command runner."""

    def run(self, spec: CommandSpec, repo_root: Path) -> CommandOutcome:
        cwd = spec.cwd or repo_root
        start = perf_counter()
        completed = subprocess.run(  # nosec B603
            list(spec.argv),
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        duration_seconds = perf_counter() - start
        return CommandOutcome(
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_seconds=duration_seconds,
        )


def _truncate(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    if len(stripped) <= _MAX_OUTPUT_CHARS:
        return stripped
    return stripped[-_MAX_OUTPUT_CHARS:]


def _command_check(
    repo_root: Path,
    runner: CommandRunner,
    spec: CommandSpec,
    *,
    evidence: str,
) -> ReadinessCheck:
    outcome = runner.run(spec, repo_root)
    status = "pass" if outcome.exit_code == 0 else "fail"
    details = (
        f"Command passed in {outcome.duration_seconds:.2f}s"
        if outcome.exit_code == 0
        else f"Command failed with exit code {outcome.exit_code} after {outcome.duration_seconds:.2f}s"
    )
    return ReadinessCheck(
        name=spec.name,
        kind="command",
        status=status,
        blocking=spec.blocking,
        details=details,
        command=spec.command,
        executed_command=list(spec.argv),
        exit_code=outcome.exit_code,
        duration_seconds=round(outcome.duration_seconds, 6),
        stdout_excerpt=_truncate(outcome.stdout),
        stderr_excerpt=_truncate(outcome.stderr),
        evidence=evidence,
    )


def _policy_check(
    name: str,
    *,
    status: str,
    details: str,
    blocking: bool,
    evidence: str,
) -> ReadinessCheck:
    return ReadinessCheck(
        name=name,
        kind="policy",
        status=status,
        blocking=blocking,
        details=details,
        evidence=evidence,
    )


def _subsystem_status(checks: Sequence[ReadinessCheck]) -> ReadinessStatus:
    if any(check.blocking and check.status != "pass" for check in checks):
        return ReadinessStatus.BLOCKED
    if any((not check.blocking) and check.status != "pass" for check in checks):
        return ReadinessStatus.ADVISORY
    return ReadinessStatus.READY


def _evaluate_static_quality(repo_root: Path, runner: CommandRunner) -> ReadinessSubsystem:
    checks = (
        _command_check(
            repo_root,
            runner,
            CommandSpec(
                name="ruff check",
                command="ruff check .",
                argv=(sys.executable, "-m", "ruff", "check", "."),
            ),
            evidence="command:ruff check .",
        ),
        _command_check(
            repo_root,
            runner,
            CommandSpec(
                name="mypy strict",
                command="mypy src --strict --config-file pyproject.toml",
                argv=(
                    sys.executable,
                    "-m",
                    "mypy",
                    "src",
                    "--strict",
                    "--config-file",
                    "pyproject.toml",
                ),
            ),
            evidence="command:mypy src --strict --config-file pyproject.toml",
        ),
        _command_check(
            repo_root,
            runner,
            CommandSpec(
                name="pylint src/bnsyn",
                command="pylint src/bnsyn",
                argv=(sys.executable, "-m", "pylint", "src/bnsyn"),
            ),
            evidence="command:pylint src/bnsyn",
        ),
    )
    return ReadinessSubsystem(
        key="static_quality",
        label="Static quality",
        status=_subsystem_status(checks),
        checks=checks,
    )


def _evaluate_runtime_proof_path(
    repo_root: Path,
    output_dir: Path,
    runner: CommandRunner,
) -> ReadinessSubsystem:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    checks = (
        _command_check(
            repo_root,
            runner,
            CommandSpec(
                name="canonical proof run",
                command="bnsyn run --profile canonical --plot --export-proof",
                argv=(
                    sys.executable,
                    "-m",
                    "bnsyn.cli",
                    "run",
                    "--profile",
                    "canonical",
                    "--plot",
                    "--export-proof",
                    "--output",
                    output_dir.as_posix(),
                ),
            ),
            evidence=f"command:bnsyn run --profile canonical --plot --export-proof --output {output_dir.as_posix()}",
        ),
    )
    return ReadinessSubsystem(
        key="runtime_proof_path",
        label="Runtime proof path",
        status=_subsystem_status(checks),
        checks=checks,
    )


def _evaluate_bundle_validation(
    repo_root: Path,
    output_dir: Path,
    runner: CommandRunner,
) -> ReadinessSubsystem:
    checks = (
        _command_check(
            repo_root,
            runner,
            CommandSpec(
                name="canonical bundle validation",
                command="bnsyn validate-bundle <artifact_dir>",
                argv=(
                    sys.executable,
                    "-m",
                    "bnsyn.cli",
                    "validate-bundle",
                    output_dir.as_posix(),
                ),
            ),
            evidence=f"command:bnsyn validate-bundle {output_dir.as_posix()}",
        ),
    )
    return ReadinessSubsystem(
        key="bundle_validation",
        label="Bundle validation",
        status=_subsystem_status(checks),
        checks=checks,
    )


def _evaluate_governance_consistency(repo_root: Path) -> ReadinessSubsystem:
    checks = (
        check_status_document_contract(repo_root / "docs" / "STATUS.md"),
        check_release_readiness_document_contract(repo_root / "docs" / "RELEASE_READINESS.md"),
        check_mutation_baseline(repo_root / "quality" / "mutation_baseline.json"),
        check_entropy_gate(repo_root),
    )
    return ReadinessSubsystem(
        key="governance_consistency",
        label="Governance consistency",
        status=_subsystem_status(checks),
        checks=checks,
    )


def _read_project_version(path: Path) -> str | None:
    if not path.exists():
        return None
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    version = data.get("project", {}).get("version")
    return str(version) if version else None


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def check_mutation_baseline(path: Path) -> ReadinessCheck:
    if not path.exists():
        return _policy_check(
            "mutation baseline",
            status="fail",
            details=f"Missing {path}",
            blocking=True,
            evidence=f"file:{path.as_posix()}",
        )

    data = _load_json(path)
    status = data.get("status")
    metrics = data.get("metrics", {})
    total_mutants = metrics.get("total_mutants")
    killed_mutants = metrics.get("killed_mutants")
    if status != "active" or not isinstance(total_mutants, int) or total_mutants <= 0:
        return _policy_check(
            "mutation baseline",
            status="fail",
            details=(
                "Mutation baseline must be active with total_mutants > 0 "
                f"(status={status!r}, total_mutants={total_mutants!r})"
            ),
            blocking=True,
            evidence=f"file:{path.as_posix()}",
        )
    if not isinstance(killed_mutants, int) or killed_mutants <= 0:
        return _policy_check(
            "mutation baseline",
            status="fail",
            details=f"metrics.killed_mutants must be > 0 (killed_mutants={killed_mutants!r})",
            blocking=True,
            evidence=f"file:{path.as_posix()}",
        )
    return _policy_check(
        "mutation baseline",
        status="pass",
        details=(
            f"Baseline active with total_mutants={total_mutants} and killed_mutants={killed_mutants}"
        ),
        blocking=True,
        evidence=f"file:{path.as_posix()}",
    )


def check_entropy_gate(repo_root: Path) -> ReadinessCheck:
    policy_path = repo_root / "entropy" / "policy.json"
    baseline_path = repo_root / "entropy" / "baseline.json"
    if not policy_path.exists() or not baseline_path.exists():
        missing = [
            candidate.as_posix()
            for candidate in (policy_path, baseline_path)
            if not candidate.exists()
        ]
        return _policy_check(
            "entropy gate",
            status="fail",
            details=f"Missing entropy inputs: {', '.join(missing)}",
            blocking=True,
            evidence="files:entropy/policy.json,entropy/baseline.json",
        )

    policy = _load_json(policy_path)
    baseline = _load_json(baseline_path)
    comparators = policy.get("comparators", {})
    if not isinstance(comparators, dict) or not comparators:
        return _policy_check(
            "entropy gate",
            status="fail",
            details="policy.json comparators missing or empty",
            blocking=True,
            evidence="file:entropy/policy.json",
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
            if current_value > baseline_value:
                failures.append(
                    f"{key}: regression (current={current_value} > baseline={baseline_value})"
                )
        elif comparator == "gte":
            if current_value < baseline_value:
                failures.append(
                    f"{key}: regression (current={current_value} < baseline={baseline_value})"
                )
        elif comparator == "eq":
            if current_value != baseline_value:
                failures.append(
                    f"{key}: changed (current={current_value} != baseline={baseline_value})"
                )
        else:
            failures.append(f"{key}: unknown comparator {comparator!r}")

    if failures:
        return _policy_check(
            "entropy gate",
            status="fail",
            details="; ".join(failures[:3]),
            blocking=True,
            evidence="files:entropy/policy.json,entropy/baseline.json",
        )

    return _policy_check(
        "entropy gate",
        status="pass",
        details="Current entropy metrics satisfy policy comparators against baseline",
        blocking=True,
        evidence="files:entropy/policy.json,entropy/baseline.json",
    )


def _status_tokens_present(text: str) -> bool:
    lowered = text.lower()
    return all(token in lowered for token in _STATUS_DOC_TOKENS)


def check_status_document_contract(path: Path) -> ReadinessCheck:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if not path.exists():
        return _policy_check(
            "STATUS readiness contract",
            status="fail",
            details=f"Missing {path}",
            blocking=True,
            evidence=f"file:{path.as_posix()}",
        )
    required_fragments = (
        "machine-readable release readiness",
        "artifacts/release_readiness.json",
        "python -m scripts.release_readiness",
    )
    missing = [fragment for fragment in required_fragments if fragment not in text]
    if not _status_tokens_present(text) or missing:
        detail_parts: list[str] = []
        if not _status_tokens_present(text):
            detail_parts.append("must define blocked/advisory/ready statuses")
        if missing:
            detail_parts.append("missing fragments: " + ", ".join(missing))
        return _policy_check(
            "STATUS readiness contract",
            status="fail",
            details="; ".join(detail_parts),
            blocking=True,
            evidence=f"file:{path.as_posix()}",
        )
    return _policy_check(
        "STATUS readiness contract",
        status="pass",
        details="STATUS.md delegates live readiness to the execution-backed report and defines all states",
        blocking=True,
        evidence=f"file:{path.as_posix()}",
    )


def check_release_readiness_document_contract(path: Path) -> ReadinessCheck:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if not path.exists():
        return _policy_check(
            "RELEASE_READINESS readiness contract",
            status="fail",
            details=f"Missing {path}",
            blocking=True,
            evidence=f"file:{path.as_posix()}",
        )
    required_fragments = (
        "static quality",
        "runtime proof path",
        "bundle validation",
        "governance consistency",
        "truth_model_version",
        "ready requires at least one execution-backed check",
    )
    missing = [fragment for fragment in required_fragments if fragment not in text.lower()]
    if not _status_tokens_present(text) or missing:
        detail_parts = []
        if not _status_tokens_present(text):
            detail_parts.append("must define blocked/advisory/ready statuses")
        if missing:
            detail_parts.append("missing fragments: " + ", ".join(missing))
        return _policy_check(
            "RELEASE_READINESS readiness contract",
            status="fail",
            details="; ".join(detail_parts),
            blocking=True,
            evidence=f"file:{path.as_posix()}",
        )
    return _policy_check(
        "RELEASE_READINESS readiness contract",
        status="pass",
        details="RELEASE_READINESS.md documents the same readiness states and execution-backed criteria",
        blocking=True,
        evidence=f"file:{path.as_posix()}",
    )
