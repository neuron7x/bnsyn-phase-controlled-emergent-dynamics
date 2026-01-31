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


def test_extract_spec_version_reads_header_version(tmp_path: Path) -> None:
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("SPECIFICATION (v2.3.4)\nDetails\n", encoding="utf-8")

    assert manifest_builder._extract_spec_version(spec_path) == "v2.3.4"


def test_extract_hypothesis_version_reads_header_version(tmp_path: Path) -> None:
    hypothesis_path = tmp_path / "HYPOTHESIS.md"
    hypothesis_path.write_text(
        "# Hypothesis\n**Version**: 2024.10\nNotes\n",
        encoding="utf-8",
    )

    assert manifest_builder._extract_hypothesis_version(hypothesis_path) == "2024.10"


def test_build_experiment_manifest_filters_manifest_json(tmp_path: Path) -> None:
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    (output_dir / "manifest.json").write_text("{}", encoding="utf-8")
    data_path = output_dir / "metrics.json"
    data_path.write_text('{"ok": true}', encoding="utf-8")

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "SPEC.md").write_text("SPEC (v1.0)\n", encoding="utf-8")
    (docs_dir / "HYPOTHESIS.md").write_text("**Version**: 0.1\n", encoding="utf-8")

    manifest = manifest_builder.build_experiment_manifest(
        output_dir=output_dir,
        experiment_name="demo",
        seeds=[1, 2],
        steps=5,
        params={"alpha": 0.1},
        repo_root=tmp_path,
    )

    expected_hash = manifest_builder._compute_file_hash(data_path)
    assert manifest["result_files"] == {"metrics.json": expected_hash}
