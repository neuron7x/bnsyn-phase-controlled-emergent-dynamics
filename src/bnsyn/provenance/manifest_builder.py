"""Manifest builders for BN-Syn CLI and experiments."""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def get_git_commit(repo_root: Path | None = None) -> str | None:
    """Get current git commit hash.

    Parameters
    ----------
    repo_root : Path | None
        Repository root for git command execution.

    Returns
    -------
    str | None
        Commit hash or None if not in git repo.
    """
    if repo_root is None:
        repo_root = _default_repo_root()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_root,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of a file.

    Parameters
    ----------
    filepath : Path
        Path to file.

    Returns
    -------
    str
        Hex digest of SHA256 hash.
    """
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        sha256.update(f.read())
    return sha256.hexdigest()


def _extract_spec_version(spec_path: Path) -> str:
    """Extract spec version from header or fallback to hash."""
    with open(spec_path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
    match = re.search(r"\((v[^)]+)\)", header)
    if match:
        return match.group(1)
    return compute_file_hash(spec_path)


def _extract_hypothesis_version(hypothesis_path: Path) -> str:
    """Extract hypothesis version from header or fallback to hash."""
    with open(hypothesis_path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(r"\*\*Version\*\*:\s*(.+)", line.strip())
            if match:
                return match.group(1).strip()
    return compute_file_hash(hypothesis_path)


def build_experiment_manifest(
    output_dir: Path,
    experiment_name: str,
    seeds: list[int],
    steps: int,
    params: dict[str, Any],
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build reproducibility manifest for experiments."""
    if repo_root is None:
        repo_root = _default_repo_root()

    result_files: dict[str, str] = {}
    for result_file in output_dir.glob("*.json"):
        if result_file.name != "manifest.json":
            result_files[result_file.name] = compute_file_hash(result_file)

    spec_path = repo_root / "docs" / "SPEC.md"
    hypothesis_path = repo_root / "docs" / "HYPOTHESIS.md"

    return {
        "experiment": experiment_name,
        "version": "1.0",
        "git_commit": get_git_commit(repo_root),
        "python_version": sys.version,
        "spec_version": _extract_spec_version(spec_path),
        "hypothesis_version": _extract_hypothesis_version(hypothesis_path),
        "spec_path": spec_path.relative_to(repo_root).as_posix(),
        "hypothesis_path": hypothesis_path.relative_to(repo_root).as_posix(),
        "seeds": seeds,
        "steps": steps,
        "params": params,
        "result_files": result_files,
    }


def build_cli_manifest(
    seed: int,
    steps_wake: int,
    steps_sleep: int,
    network_size: int,
    package_version: str,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build manifest for CLI output."""
    manifest: dict[str, Any] = {
        "seed": seed,
        "steps_wake": steps_wake,
        "steps_sleep": steps_sleep,
        "N": network_size,
        "package_version": package_version,
    }
    git_sha = get_git_commit(repo_root)
    if git_sha:
        manifest["git_sha"] = git_sha
    return manifest
