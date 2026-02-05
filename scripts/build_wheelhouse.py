from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version


@dataclass(frozen=True)
class TargetConfig:
    python_version: str
    implementation: str
    abi: str
    platform_tag: str


@dataclass(frozen=True)
class LockedRequirement:
    raw: str
    name: str
    version: str
    marker: str | None

    @property
    def key(self) -> tuple[str, Version]:
        return canonicalize_name(self.name), Version(self.version)


@dataclass(frozen=True)
class ParseResult:
    requirements: list[LockedRequirement]
    unsupported: list[str]


def _normalize_python_full_version(python_version: str) -> str:
    parts = python_version.split(".")
    if len(parts) == 2:
        return f"{python_version}.0"
    return python_version


def _marker_environment(target: TargetConfig) -> dict[str, str]:
    env = default_environment()
    env["python_version"] = target.python_version
    env["python_full_version"] = _normalize_python_full_version(target.python_version)
    env["implementation_name"] = (
        "cpython" if target.implementation == "cp" else target.implementation
    )
    return env


def _iter_requirement_lines(lock_file: Path) -> list[str]:
    lines: list[str] = []
    buffer = ""
    for raw_line in lock_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            if buffer:
                lines.append(buffer.strip())
                buffer = ""
            continue
        if line.startswith("--hash="):
            if buffer:
                lines.append(buffer.strip())
                buffer = ""
            continue
        if line.startswith("--"):
            continue

        if line.endswith("\\"):
            buffer += line[:-1].strip() + " "
            continue

        buffer += line
        lines.append(buffer.strip())
        buffer = ""

    if buffer:
        lines.append(buffer.strip())
    return lines


def parse_locked_requirements(lock_file: Path, target: TargetConfig) -> ParseResult:
    applicable: list[LockedRequirement] = []
    unsupported: list[str] = []
    marker_env = _marker_environment(target)

    for req_line in _iter_requirement_lines(lock_file):
        try:
            req = Requirement(req_line)
        except InvalidRequirement:
            unsupported.append(req_line)
            continue

        if len(req.specifier) != 1:
            unsupported.append(req_line)
            continue

        spec = next(iter(req.specifier))
        if spec.operator != "==" or "*" in spec.version:
            unsupported.append(req_line)
            continue

        if req.marker is not None and not req.marker.evaluate(marker_env):
            continue

        applicable.append(
            LockedRequirement(
                raw=req_line,
                name=req.name,
                version=spec.version,
                marker=str(req.marker) if req.marker is not None else None,
            )
        )

    return ParseResult(requirements=applicable, unsupported=unsupported)


def wheelhouse_coverage(wheelhouse_dir: Path) -> dict[tuple[str, Version], set[str]]:
    coverage: dict[tuple[str, Version], set[str]] = {}
    for wheel_path in wheelhouse_dir.glob("*.whl"):
        try:
            parsed_name, parsed_version, _, _ = parse_wheel_filename(wheel_path.name)
        except (InvalidVersion, ValueError):
            continue
        key = (canonicalize_name(parsed_name), parsed_version)
        coverage.setdefault(key, set()).add(wheel_path.name)
    return coverage


def _build_report(
    lock_file: Path,
    wheelhouse_dir: Path,
    target: TargetConfig,
    parsed: ParseResult,
) -> dict[str, Any]:
    coverage = wheelhouse_coverage(wheelhouse_dir)
    missing: list[str] = []
    for requirement in parsed.requirements:
        if requirement.key not in coverage:
            missing.append(f"{requirement.name}=={requirement.version}")

    wheels_by_requirement = {
        f"{name}=={str(version)}": sorted(files)
        for (name, version), files in sorted(
            coverage.items(), key=lambda x: (x[0][0], str(x[0][1]))
        )
    }

    return {
        "lock_file": str(lock_file),
        "wheelhouse_dir": str(wheelhouse_dir),
        "target": {
            "python_version": target.python_version,
            "implementation": target.implementation,
            "abi": target.abi,
            "platform_tag": target.platform_tag,
        },
        "parsed_requirements_count": len(_iter_requirement_lines(lock_file)),
        "applicable_requirements_count": len(parsed.requirements),
        "unsupported_requirements": sorted(parsed.unsupported),
        "missing": sorted(missing),
        "wheel_inventory_count": sum(len(files) for files in coverage.values()),
        "wheel_inventory": wheels_by_requirement,
    }


def _write_report(report_path: Path | None, report: dict[str, Any]) -> None:
    if report_path is None:
        return
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_wheelhouse(
    lock_file: Path,
    wheelhouse_dir: Path,
    target: TargetConfig,
    report_path: Path | None = None,
) -> int:
    parsed = parse_locked_requirements(lock_file, target)
    report = _build_report(
        lock_file=lock_file, wheelhouse_dir=wheelhouse_dir, target=target, parsed=parsed
    )
    _write_report(report_path, report)

    if report["unsupported_requirements"]:
        print(
            "Unsupported lock entries detected (only pinned '==' entries are allowed):",
            file=sys.stderr,
        )
        for entry in report["unsupported_requirements"]:
            print(f"  - {entry}", file=sys.stderr)
        return 2

    if report["missing"]:
        print("Missing wheel artifacts for locked dependencies:", file=sys.stderr)
        for requirement in report["missing"]:
            print(f"  - {requirement}", file=sys.stderr)
        return 1

    print("wheelhouse validation passed: all applicable locked dependencies are covered.")
    return 0


def build_wheelhouse(lock_file: Path, wheelhouse_dir: Path, target: TargetConfig) -> None:
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
        target.python_version,
        "--implementation",
        target.implementation,
        "--abi",
        target.abi,
        "--platform",
        target.platform_tag,
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            "wheelhouse build failed: one or more locked dependencies are unavailable as wheels "
            f"for target python={target.python_version}, implementation={target.implementation}, "
            f"abi={target.abi}, platform={target.platform_tag}."
        ) from error


def _default_target(python_version: str) -> TargetConfig:
    normalized = python_version.replace(".", "")
    return TargetConfig(
        python_version=python_version,
        implementation="cp",
        abi=f"cp{normalized}",
        platform_tag=sysconfig_platform_tag(),
    )


def sysconfig_platform_tag() -> str:
    machine = platform.machine().replace("-", "_").replace(".", "_")
    system = platform.system().lower()
    if system == "linux":
        return f"manylinux_2_17_{machine}"
    if system == "darwin":
        return f"macosx_11_0_{machine}"
    if system == "windows":
        if machine in {"x86_64", "amd64"}:
            return "win_amd64"
        return f"win_{machine}"
    return "any"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and validate offline wheelhouse artifacts.")
    parser.add_argument("--lock-file", default="requirements-lock.txt", type=Path)
    parser.add_argument("--wheelhouse", default="wheelhouse", type=Path)
    parser.add_argument("--python-version", default="3.11")
    parser.add_argument("--implementation", default=None)
    parser.add_argument("--abi", default=None)
    parser.add_argument("--platform-tag", default=None)
    parser.add_argument("--report", type=Path, default=None)

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build", help="Download all pinned wheels for the configured target.")
    subparsers.add_parser(
        "validate", help="Validate wheelhouse covers all applicable pinned dependencies."
    )

    args = parser.parse_args()

    target = _default_target(args.python_version)
    if args.implementation is not None:
        target = TargetConfig(
            python_version=target.python_version,
            implementation=args.implementation,
            abi=target.abi,
            platform_tag=target.platform_tag,
        )
    if args.abi is not None:
        target = TargetConfig(
            python_version=target.python_version,
            implementation=target.implementation,
            abi=args.abi,
            platform_tag=target.platform_tag,
        )
    if args.platform_tag is not None:
        target = TargetConfig(
            python_version=target.python_version,
            implementation=target.implementation,
            abi=target.abi,
            platform_tag=args.platform_tag,
        )

    if args.command == "build":
        build_wheelhouse(args.lock_file, args.wheelhouse, target)
        return 0

    return validate_wheelhouse(
        lock_file=args.lock_file,
        wheelhouse_dir=args.wheelhouse,
        target=target,
        report_path=args.report,
    )


if __name__ == "__main__":
    raise SystemExit(main())
