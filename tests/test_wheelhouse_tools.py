from __future__ import annotations

import json
from pathlib import Path

from scripts.build_wheelhouse import TargetConfig, parse_locked_requirements, validate_wheelhouse


def _linux_target() -> TargetConfig:
    return TargetConfig(
        python_version="3.11",
        implementation="cp",
        abi="cp311",
        platform_tag="manylinux_2_17_x86_64",
        sys_platform="linux",
        os_name="posix",
        platform_system="Linux",
        platform_machine="x86_64",
    )


def _win_target() -> TargetConfig:
    return TargetConfig(
        python_version="3.11",
        implementation="cp",
        abi="cp311",
        platform_tag="win_amd64",
        sys_platform="win32",
        os_name="nt",
        platform_system="Windows",
        platform_machine="amd64",
    )


def test_parse_locked_requirements_ignores_hashes_comments_and_continuations(tmp_path: Path) -> None:
    lock_file = tmp_path / "requirements-lock.txt"
    lock_file.write_text(
        (
            "# generated\n"
            "numpy==1.26.4 \\\n"
            "    --hash=sha256:abc\n"
            "packaging==24.2\n"
            "    --hash=sha256:def\n"
        ),
        encoding="utf-8",
    )

    parsed = parse_locked_requirements(lock_file, _linux_target())

    assert [(r.name, r.version) for r in parsed.requirements] == [
        ("numpy", "1.26.4"),
        ("packaging", "24.2"),
    ]
    assert parsed.unsupported == []


def test_parse_locked_requirements_evaluates_markers_against_target_not_host(tmp_path: Path) -> None:
    lock_file = tmp_path / "requirements-lock.txt"
    lock_file.write_text(
        "\n".join(
            [
                'linuxonly==1.0.0; sys_platform == "linux"',
                'windowsonly==2.0.0; sys_platform == "win32"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    parsed_linux = parse_locked_requirements(lock_file, _linux_target())
    assert [(r.name, r.version) for r in parsed_linux.requirements] == [("linuxonly", "1.0.0")]

    parsed_win = parse_locked_requirements(lock_file, _win_target())
    assert [(r.name, r.version) for r in parsed_win.requirements] == [("windowsonly", "2.0.0")]


def test_validate_wheelhouse_reports_missing_locked_wheels(tmp_path: Path) -> None:
    lock_file = tmp_path / "requirements-lock.txt"
    lock_file.write_text("numpy==1.26.4\npackaging==24.2\n", encoding="utf-8")

    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    (wheelhouse / "numpy-1.26.4-cp311-cp311-manylinux_2_17_x86_64.whl").write_bytes(b"")

    exit_code = validate_wheelhouse(lock_file, wheelhouse, _linux_target())
    assert exit_code == 1


def test_validate_wheelhouse_fails_on_unsupported_lock_entries(tmp_path: Path) -> None:
    lock_file = tmp_path / "requirements-lock.txt"
    lock_file.write_text("unpinned>=1.0.0\n", encoding="utf-8")

    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()

    exit_code = validate_wheelhouse(lock_file, wheelhouse, _linux_target())
    assert exit_code == 2


def test_validate_wheelhouse_writes_deterministic_report(tmp_path: Path) -> None:
    lock_file = tmp_path / "requirements-lock.txt"
    lock_file.write_text("numpy==1.26.4\n", encoding="utf-8")

    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    (wheelhouse / "numpy-1.26.4-cp311-cp311-manylinux_2_17_x86_64.whl").write_bytes(b"")

    report_path = tmp_path / "artifacts" / "wheelhouse_report.json"
    exit_code = validate_wheelhouse(lock_file, wheelhouse, _linux_target(), report_path=report_path)

    assert exit_code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert sorted(report.keys()) == [
        "applicable_requirements_count",
        "lock_file",
        "missing",
        "parsed_requirements_count",
        "target",
        "unsupported_requirements",
        "wheel_inventory",
        "wheel_inventory_count",
        "wheelhouse_dir",
    ]
    assert report["parsed_requirements_count"] == 1
    assert report["applicable_requirements_count"] == 1
    assert report["missing"] == []
    assert report["unsupported_requirements"] == []
    assert "numpy==1.26.4" in report["wheel_inventory"]
