#!/usr/bin/env python3
"""Enforce quickstart setup/demo/test contract consistency."""

from __future__ import annotations

import json
import re
import subprocess
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


def _extract_readme_quickstart(text: str) -> list[str]:
    lines = text.splitlines()
    heading_index: int | None = None
    for index, line in enumerate(lines):
        if line.strip() == "## Quickstart":
            heading_index = index
            break
    if heading_index is None:
        raise ContractError("README.md missing '## Quickstart' heading")

    fence_start: int | None = None
    for index in range(heading_index + 1, len(lines)):
        stripped = lines[index].strip()
        if stripped.startswith("## "):
            break
        if stripped.startswith("```"):
            fence_start = index
            break
    if fence_start is None:
        raise ContractError("README.md Quickstart section missing fenced code block")

    fence_header = lines[fence_start].strip()
    if fence_header not in {"```", "```bash", "```sh", "```shell"}:
        raise ContractError("README.md Quickstart first fenced block must be a shell code block")

    block_lines: list[str] = []
    for index in range(fence_start + 1, len(lines)):
        stripped = lines[index].strip()
        if stripped == "```":
            return [line for line in block_lines if line]
        block_lines.append(stripped)

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


def _run_with_timeout(command: list[str], timeout_seconds: int) -> subprocess.CompletedProcess[str]:
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
        raise ContractError(f"command timed out after {timeout_seconds}s: {joined}") from exc


def _count_collected_tests(output: str) -> int | None:
    aggregate = 0
    found_aggregate_lines = False
    for raw in output.splitlines():
        line = raw.strip()
        match = re.match(r"^[^:]+\.py:\s+(\d+)$", line)
        if match:
            aggregate += int(match.group(1))
            found_aggregate_lines = True
    if found_aggregate_lines:
        return aggregate

    summary_patterns = (
        r"(\d+)(?:/\d+)?\s+tests?\s+collected",
        r"collected\s+(\d+)\s+items?",
    )
    for pattern in summary_patterns:
        match = re.search(pattern, output)
        if match:
            return int(match.group(1))
    return None


def _assert_non_empty_gate_suite() -> None:
    proc = _run_with_timeout(
        ["python", "-m", "pytest", "--collect-only", "-q", "-m", GATE_MARKER],
        timeout_seconds=COLLECT_TIMEOUT_SECONDS,
    )
    collected = _count_collected_tests(proc.stdout)
    if collected is None:
        raise ContractError("Unable to determine collected test count for gate suite")
    if collected <= 0:
        raise ContractError("Gate suite collected 0 tests")


def _assert_demo_artifact() -> None:
    _run_with_timeout(["make", "demo"], timeout_seconds=DEMO_TIMEOUT_SECONDS)
    with ARTIFACT_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ContractError("artifacts/demo.json must be a JSON object")
    demo = payload.get("demo")
    if not isinstance(demo, dict) or not demo:
        raise ContractError("artifacts/demo.json must contain a non-empty 'demo' object")


def main() -> int:
    readme_text = README_PATH.read_text(encoding="utf-8")
    makefile_text = MAKEFILE_PATH.read_text(encoding="utf-8")

    quickstart_lines = _extract_readme_quickstart(readme_text)
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
