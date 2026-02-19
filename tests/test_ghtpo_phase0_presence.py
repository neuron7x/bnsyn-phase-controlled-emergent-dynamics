from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


REQUIRED_FILES = [
    ".github/ghtpo.yaml",
    ".github/pull_request_template.md",
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/feature_request.yml",
    "CONTRIBUTING.md",
    "SECURITY.md",
    ".github/CODEOWNERS",
    ".github/dependabot.yml",
]


def test_phase0_files_exist_and_non_empty() -> None:
    for rel in REQUIRED_FILES:
        path = REPO_ROOT / rel
        assert path.exists(), f"Missing required file: {rel}"
        assert path.is_file(), f"Expected file: {rel}"
        assert path.stat().st_size > 0, f"Required file is empty: {rel}"
