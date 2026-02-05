from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

_PINNED_REQUIREMENT = re.compile(r"^([A-Za-z0-9_.-]+)==([^\\\s;]+)")


@dataclass(frozen=True)
class LockedRequirement:
    name: str
    version: str

    @property
    def key(self) -> tuple[str, Version]:
        return canonicalize_name(self.name), Version(self.version)


def parse_locked_requirements(lock_file: Path) -> list[LockedRequirement]:
    requirements: list[LockedRequirement] = []
    for raw_line in lock_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PINNED_REQUIREMENT.match(line)
        if match is None:
            continue
        requirements.append(LockedRequirement(name=match.group(1), version=match.group(2)))
    return requirements


def wheelhouse_coverage(wheelhouse_dir: Path) -> set[tuple[str, Version]]:
    coverage: set[tuple[str, Version]] = set()
    for wheel_path in wheelhouse_dir.glob("*.whl"):
        parsed_name, parsed_version, _, _ = parse_wheel_filename(wheel_path.name)
        coverage.add((canonicalize_name(parsed_name), parsed_version))
    return coverage


def validate_wheelhouse(lock_file: Path, wheelhouse_dir: Path) -> list[LockedRequirement]:
    missing: list[LockedRequirement] = []
    coverage = wheelhouse_coverage(wheelhouse_dir)
    for requirement in parse_locked_requirements(lock_file):
        if requirement.key not in coverage:
            missing.append(requirement)
    return missing


def build_wheelhouse(lock_file: Path, wheelhouse_dir: Path, python_version: str) -> None:
    wheelhouse_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--only-binary=:all:",
        "--dest",
        str(wheelhouse_dir),
        "--requirement",
        str(lock_file),
        "--python-version",
        python_version,
        "--implementation",
        "cp",
        "--abi",
        f"cp{python_version.replace('.', '')}",
    ]
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and validate offline wheelhouse artifacts.")
    parser.add_argument("--lock-file", default="requirements-lock.txt", type=Path)
    parser.add_argument("--wheelhouse", default="wheelhouse", type=Path)
    parser.add_argument("--python-version", default="3.11")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build", help="Download all pinned wheels for the configured Python version.")
    subparsers.add_parser("validate", help="Validate that wheelhouse covers all pinned dependencies.")

    args = parser.parse_args()

    if args.command == "build":
        build_wheelhouse(args.lock_file, args.wheelhouse, args.python_version)
        return 0

    missing = validate_wheelhouse(args.lock_file, args.wheelhouse)
    if missing:
        print("Missing wheel artifacts for locked dependencies:", file=sys.stderr)
        for requirement in missing:
            print(f"  - {requirement.name}=={requirement.version}", file=sys.stderr)
        return 1

    print("wheelhouse validation passed: all locked dependencies are covered.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
