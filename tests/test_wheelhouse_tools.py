from __future__ import annotations

from pathlib import Path

from scripts.build_wheelhouse import parse_locked_requirements, validate_wheelhouse


def test_parse_locked_requirements_reads_pinned_packages(tmp_path: Path) -> None:
    lock_file = tmp_path / "requirements-lock.txt"
    lock_file.write_text(
        """
# generated
numpy==1.26.4 \\
    --hash=sha256:abc
# comment
packaging==24.2
    --hash=sha256:def
""".strip()
        + "\n",
        encoding="utf-8",
    )

    parsed = parse_locked_requirements(lock_file)

    assert [(item.name, item.version) for item in parsed] == [
        ("numpy", "1.26.4"),
        ("packaging", "24.2"),
    ]


def test_validate_wheelhouse_reports_missing_locked_wheels(tmp_path: Path) -> None:
    lock_file = tmp_path / "requirements-lock.txt"
    lock_file.write_text("numpy==1.26.4\npackaging==24.2\n", encoding="utf-8")

    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    (wheelhouse / "numpy-1.26.4-cp311-cp311-manylinux_2_17_x86_64.whl").write_bytes(b"")

    missing = validate_wheelhouse(lock_file, wheelhouse)

    assert [(item.name, item.version) for item in missing] == [("packaging", "24.2")]
