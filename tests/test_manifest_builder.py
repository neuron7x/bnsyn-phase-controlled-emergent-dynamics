from __future__ import annotations

import subprocess
from pathlib import Path

from bnsyn.provenance import manifest_builder


def test_get_git_commit_handles_errors(tmp_path: Path, monkeypatch) -> None:
    def raise_called(*args, **kwargs) -> None:
        raise subprocess.CalledProcessError(1, ["git", "rev-parse", "HEAD"])

    monkeypatch.setattr(manifest_builder.subprocess, "run", raise_called)
    assert manifest_builder._get_git_commit(tmp_path) is None

    def raise_missing(*args, **kwargs) -> None:
        raise FileNotFoundError("git not available")

    monkeypatch.setattr(manifest_builder.subprocess, "run", raise_missing)
    assert manifest_builder._get_git_commit(tmp_path) is None


def test_extract_spec_version_falls_back_to_hash(tmp_path: Path) -> None:
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("Spec header without version\nMore\n", encoding="utf-8")

    expected = manifest_builder._compute_file_hash(spec_path)

    assert manifest_builder._extract_spec_version(spec_path) == expected


def test_extract_hypothesis_version_falls_back_to_hash(tmp_path: Path) -> None:
    hypothesis_path = tmp_path / "HYPOTHESIS.md"
    hypothesis_path.write_text("# Hypothesis\nNo version here\n", encoding="utf-8")

    expected = manifest_builder._compute_file_hash(hypothesis_path)

    assert manifest_builder._extract_hypothesis_version(hypothesis_path) == expected
