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


class ContractError(RuntimeError):
    """Quickstart contract violation."""


def _extract_readme_quickstart(text: str) -> list[str]:
    pattern = re.compile(
        r"^## Quickstart\n\n```bash\n(?P<body>.*?)\n```",
        flags=re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        raise ContractError("README.md missing Quickstart code block")
    return [line.strip() for line in match.group("body").strip().splitlines() if line.strip()]


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


def _assert_non_empty_gate_suite() -> None:
    proc = subprocess.run(
        ["python", "-m", "pytest", "--collect-only", "-m", GATE_MARKER],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"(\d+)(?:/\d+)? tests collected", proc.stdout)
    if match is None:
        raise ContractError("Unable to determine collected test count for gate suite")
    if int(match.group(1)) <= 0:
        raise ContractError("Gate suite collected 0 tests")


def _assert_demo_artifact() -> None:
    subprocess.run(["make", "demo"], check=True)
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
