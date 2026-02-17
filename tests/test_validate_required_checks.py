from __future__ import annotations

from pathlib import Path

import pytest

from scripts.validate_required_checks import RequiredChecksParseError, load_required_checks, main


def test_load_required_checks_rejects_duplicate_workflow_file(tmp_path: Path) -> None:
    path = tmp_path / "REQUIRED_CHECKS.yml"
    path.write_text(
        "\n".join(
            [
                "version: '1'",
                "required_checks:",
                "  - workflow_file: ci-pr-atomic.yml",
                "  - workflow_file: ci-pr-atomic.yml",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(RequiredChecksParseError, match="Duplicate workflow_file"):
        load_required_checks(path)


def test_load_required_checks_rejects_non_mapping_yaml_root(tmp_path: Path) -> None:
    path = tmp_path / "REQUIRED_CHECKS.yml"
    path.write_text("- foo\n- bar\n", encoding="utf-8")

    with pytest.raises(RequiredChecksParseError, match="must be a mapping"):
        load_required_checks(path)


def test_main_supports_help_flag(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["validate_required_checks", "--help"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Usage: python -m scripts.validate_required_checks" in captured.out
