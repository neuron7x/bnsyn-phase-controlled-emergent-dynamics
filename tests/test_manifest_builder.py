from __future__ import annotations

import subprocess
from importlib import metadata
from pathlib import Path

import pytest

from bnsyn.provenance import manifest_builder


def test_get_git_commit_handles_errors(tmp_path: Path, monkeypatch) -> None:
    try:
        version = metadata.version("bnsyn")
    except metadata.PackageNotFoundError:
        version = "0.0.0"

    def raise_called(*args, **kwargs) -> None:
        raise subprocess.CalledProcessError(1, ["git", "rev-parse", "HEAD"])

    monkeypatch.setattr(manifest_builder.subprocess, "run", raise_called)
    commit = manifest_builder._get_git_commit(tmp_path)
    assert commit.startswith(f"release-{version}+nogit.")

    def raise_missing(*args, **kwargs) -> None:
        raise FileNotFoundError("git not available")

    monkeypatch.setattr(manifest_builder.subprocess, "run", raise_missing)
    commit = manifest_builder._get_git_commit(tmp_path)
    assert commit.startswith(f"release-{version}+nogit.")




def test_build_lineage_fallback_is_directory_invariant(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "pyproject.toml").write_text(
        "[project]\nname='bnsyn'\nversion='1.2.3'\n", encoding="utf-8"
    )
    nested = repo_root / "a" / "b"
    nested.mkdir(parents=True)

    first = manifest_builder._build_lineage_fallback(repo_root, "1.2.3")
    second = manifest_builder._build_lineage_fallback(nested, "1.2.3")

    assert first == second

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


def test_resolve_package_version_fallback_from_pyproject(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='bnsyn'\nversion='9.9.9'\n", encoding="utf-8"
    )

    def raise_pkg_not_found(_: str) -> str:
        raise metadata.PackageNotFoundError

    monkeypatch.setattr(manifest_builder.metadata, "version", raise_pkg_not_found)
    assert manifest_builder._resolve_package_version(tmp_path) == "9.9.9"


def test_resolve_package_version_invalid_toml_returns_default(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "pyproject.toml").write_text("{ this is invalid toml", encoding="utf-8")

    def raise_pkg_not_found(_: str) -> str:
        raise metadata.PackageNotFoundError

    monkeypatch.setattr(manifest_builder.metadata, "version", raise_pkg_not_found)
    assert manifest_builder._resolve_package_version(tmp_path) == "0.0.0"


def test_build_sleep_stack_manifest_uses_fallback_git_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(manifest_builder.shutil, "which", lambda _: None)

    manifest = manifest_builder.build_sleep_stack_manifest(
        seed=1,
        steps_wake=2,
        steps_sleep=3,
        N=4,
        package_version="0.2.0",
        repo_root=tmp_path,
    )

    assert manifest["git_sha"].startswith("release-0.2.0+nogit.")
