"""Tests for CLI and experiment manifest consistency."""

from __future__ import annotations

from pathlib import Path

from bnsyn.provenance.manifest_builder import build_cli_manifest, build_experiment_manifest


def test_manifest_git_consistency(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "results.json").write_text("{}", encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[1]
    experiment_manifest = build_experiment_manifest(
        output_dir=output_dir,
        experiment_name="temp_ablation_v1",
        seeds=[0],
        steps=10,
        params={},
        repo_root=repo_root,
    )
    cli_manifest = build_cli_manifest(
        seed=0,
        steps_wake=10,
        steps_sleep=10,
        network_size=40,
        package_version="0.2.0",
        repo_root=repo_root,
    )

    git_commit = experiment_manifest["git_commit"]
    if git_commit is None:
        assert "git_sha" not in cli_manifest
    else:
        assert cli_manifest.get("git_sha") == git_commit
