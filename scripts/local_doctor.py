#!/usr/bin/env python3
"""Local Linux environment verifier for canonical proof execution."""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    remediation: str | None = None


def _is_supported_python(version: tuple[int, int, int]) -> bool:
    return version >= (3, 11, 0)


def _check_linux() -> CheckResult:
    if platform.system() == "Linux":
        return CheckResult("linux", True, "Linux host detected.")
    return CheckResult(
        "linux",
        False,
        f"Unsupported OS: {platform.system()}.",
        "Use Ubuntu/Linux for the supported local deployment path.",
    )


def _check_python() -> CheckResult:
    ver = sys.version_info[:3]
    if _is_supported_python(ver):
        return CheckResult("python", True, f"Python {ver[0]}.{ver[1]}.{ver[2]} is supported (>=3.11).")
    return CheckResult(
        "python",
        False,
        f"Python {ver[0]}.{ver[1]}.{ver[2]} is unsupported.",
        "Install Python 3.11+ and re-run: ./scripts/bootstrap_local_linux.sh",
    )


def _check_venv(repo_root: Path) -> CheckResult:
    venv_dir = repo_root / ".venv"
    if os.environ.get("VIRTUAL_ENV"):
        return CheckResult("venv", True, f"Active virtualenv: {os.environ['VIRTUAL_ENV']}")
    if (venv_dir / "bin" / "python").exists():
        return CheckResult(
            "venv",
            True,
            "Project virtualenv found at .venv (not active, scripts will still use it).",
        )
    return CheckResult(
        "venv",
        False,
        "No active virtualenv and .venv is missing.",
        "Run: ./scripts/bootstrap_local_linux.sh",
    )


def _check_bnsyn_install() -> CheckResult:
    try:
        importlib.import_module("bnsyn")
    except Exception:
        return CheckResult(
            "bnsyn-install",
            False,
            "Package 'bnsyn' is not importable in this interpreter.",
            "Run: python -m pip install -e '.[plot]'",
        )
    return CheckResult("bnsyn-install", True, "Package 'bnsyn' is importable.")


def _check_plot_dependencies() -> CheckResult:
    try:
        importlib.import_module("matplotlib")
    except Exception:
        return CheckResult(
            "plot-deps",
            False,
            "matplotlib is missing, but canonical command includes --plot.",
            "Run: python -m pip install -e '.[plot]'",
        )
    return CheckResult("plot-deps", True, "matplotlib import check passed.")


def _check_cli_path() -> CheckResult:
    if shutil.which("bnsyn"):
        return CheckResult("cli", True, "bnsyn CLI is available on PATH.")
    return CheckResult(
        "cli",
        False,
        "bnsyn CLI is not on PATH.",
        "Use .venv binaries directly or activate env: source .venv/bin/activate",
    )


def run_checks(repo_root: Path) -> list[CheckResult]:
    return [
        _check_linux(),
        _check_python(),
        _check_venv(repo_root),
        _check_bnsyn_install(),
        _check_plot_dependencies(),
        _check_cli_path(),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate local environment for canonical BN-Syn proof run.")
    parser.add_argument("--strict", action="store_true", help="Fail if CLI is not on PATH.")
    args = parser.parse_args()

    results = run_checks(Path.cwd())
    hard_failures = {"linux", "python", "venv", "bnsyn-install", "plot-deps"}
    if args.strict:
        hard_failures.add("cli")

    failed = False
    print("Local environment doctor report:")
    for result in results:
        prefix = "PASS" if result.ok else "FAIL"
        print(f" - [{prefix}] {result.name}: {result.detail}")
        if not result.ok and result.remediation:
            print(f"   Fix: {result.remediation}")
        if not result.ok and result.name in hard_failures:
            failed = True

    if failed:
        print("\nEnvironment verification failed. Apply fixes above and rerun.")
        return 1

    print("\nEnvironment verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
