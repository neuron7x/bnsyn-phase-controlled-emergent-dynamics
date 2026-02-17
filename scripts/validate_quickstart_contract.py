#!/usr/bin/env python3
"""Enforce quickstart setup/demo/test contract consistency."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

README_PATH = Path("README.md")
MAKEFILE_PATH = Path("Makefile")
ARTIFACT_PATH = Path("artifacts/demo.json")
QUICKSTART_LINES = ["make setup", "make demo", "make test"]
CANONICAL_TEST_CMD = 'python -m pytest -m "not (validation or property)" -q'
GATE_MARKER = "not (validation or property)"
DEMO_TIMEOUT_SECONDS = 120
COLLECT_TIMEOUT_SECONDS = 120


class ContractError(RuntimeError):
    """Quickstart contract violation."""


def _tail_40_lines(text: str) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-40:])


def extract_quickstart_lines(readme_text: str) -> list[str]:
    """Extract non-empty lines from first fenced code block after ## Quickstart."""
    lines = readme_text.splitlines()

    heading_index: int | None = None
    for idx, line in enumerate(lines):
        if line.strip() == "## Quickstart":
            heading_index = idx
            break
    if heading_index is None:
        raise ContractError("README.md missing '## Quickstart' heading")

    fence_start: int | None = None
    for idx in range(heading_index + 1, len(lines)):
        stripped = lines[idx].strip()
        if stripped.startswith("## "):
            break
        if stripped in {"```", "```bash", "```sh", "```shell"}:
            fence_start = idx
            break
    if fence_start is None:
        raise ContractError("README.md Quickstart section missing fenced code block")

    extracted: list[str] = []
    for idx in range(fence_start + 1, len(lines)):
        stripped = lines[idx].strip()
        if stripped == "```":
            return [line for line in extracted if line]
        extracted.append(stripped)

    raise ContractError("README.md Quickstart fenced code block is not closed")


def _assert_make_targets(makefile_text: str) -> None:
    for target in ("setup", "demo", "test"):
        if re.search(rf"(?m)^{target}:", makefile_text) is None:
            raise ContractError(f"Makefile missing required target: {target}")


def _assert_test_contract(makefile_text: str) -> None:
    if 'test:\n\t$(MAKE) test-gate' not in makefile_text:
        raise ContractError("Makefile test target must delegate to make test-gate")
    if f"TEST_CMD ?= {CANONICAL_TEST_CMD}" not in makefile_text:
        raise ContractError("Makefile TEST_CMD is not canonical")
    if "test-gate:\n\t$(TEST_CMD)" not in makefile_text:
        raise ContractError("Makefile test-gate must execute $(TEST_CMD)")


def _run_checked(command: list[str], timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        joined = " ".join(command)
        raise ContractError(f"timeout {timeout_seconds}s: {joined}") from exc
    except subprocess.CalledProcessError as exc:
        joined = " ".join(command)
        combined = (exc.stdout or "") + "\n" + (exc.stderr or "")
        trimmed = _tail_40_lines(combined).strip()
        raise ContractError(f"command failed rc={exc.returncode}: {joined} | {trimmed}") from exc


def count_gate_collected(stdout: str) -> int | None:
    """Count collected gate tests robustly across pytest output variants."""
    nodeid_count = 0
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line or line.startswith(("=", "WARNING", "DeprecationWarning")):
            continue
        if "::" in line:
            nodeid_count += 1
    if nodeid_count > 0:
        return nodeid_count

    for raw in stdout.splitlines():
        line = raw.strip()
        match = re.match(r"^[^:]+\.py:\s+(\d+)$", line)
        if match:
            return int(match.group(1))

    patterns = (r"collected\s+(\d+)\s+items?", r"(\d+)\s+tests?\s+collected")
    for pattern in patterns:
        match = re.search(pattern, stdout)
        if match:
            return int(match.group(1))
    return None


def _assert_non_empty_gate_suite() -> None:
    proc = _run_checked(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-m",
            GATE_MARKER,
            "--disable-warnings",
        ],
        timeout_seconds=COLLECT_TIMEOUT_SECONDS,
    )
    collected = count_gate_collected(proc.stdout)
    if collected is None:
        raise ContractError("Unable to determine collected test count for gate suite")
    if collected == 0:
        raise ContractError("Gate suite collected 0 tests")


def _assert_demo_artifact() -> None:
    _run_checked(["make", "demo"], timeout_seconds=DEMO_TIMEOUT_SECONDS)
    if not ARTIFACT_PATH.exists():
        raise ContractError("artifacts/demo.json was not created by make demo")
    try:
        with ARTIFACT_PATH.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ContractError("artifacts/demo.json is not valid JSON") from exc
    except OSError as exc:
        raise ContractError(f"cannot read artifacts/demo.json: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError("artifacts/demo.json must be a JSON object")
    demo = payload.get("demo")
    if not isinstance(demo, dict) or not demo:
        raise ContractError("artifacts/demo.json must contain a non-empty 'demo' object")


def main() -> int:
    try:
        readme_text = README_PATH.read_text(encoding="utf-8")
        makefile_text = MAKEFILE_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"required file access failed: {exc}") from exc

    quickstart_lines = extract_quickstart_lines(readme_text)
    if quickstart_lines != QUICKSTART_LINES:
        raise ContractError(
            "README Quickstart block must contain exactly: " + ", ".join(QUICKSTART_LINES)
        )

    _assert_make_targets(makefile_text)
    _assert_test_contract(makefile_text)
    _assert_demo_artifact()
    _assert_non_empty_gate_suite()

    print("quickstart contract validation PASSED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"quickstart contract validation FAILED: {exc}")
        raise SystemExit(1)
