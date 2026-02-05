from __future__ import annotations

from pathlib import Path

from scripts.lint_ci_truthfulness import WorkflowLinter


def test_flags_job_continue_on_error(tmp_path: Path) -> None:
    workflows_dir = tmp_path / ".github" / "workflows"
    workflows_dir.mkdir(parents=True)
    (workflows_dir / "ci.yml").write_text(
        """
name: CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    continue-on-error: true
    steps:
      - name: Run tests
        run: pytest -q
""".strip(),
        encoding="utf-8",
    )

    result = WorkflowLinter().lint_all(workflows_dir)

    assert any(v.category == "Masked Job Failure" for v in result.violations)


def test_flags_step_continue_on_error(tmp_path: Path) -> None:
    workflows_dir = tmp_path / ".github" / "workflows"
    workflows_dir.mkdir(parents=True)
    (workflows_dir / "ci.yml").write_text(
        """
name: CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Run tests
        run: pytest -q
        continue-on-error: true
""".strip(),
        encoding="utf-8",
    )

    result = WorkflowLinter().lint_all(workflows_dir)

    assert any(v.category == "Masked Step Failure" for v in result.violations)
