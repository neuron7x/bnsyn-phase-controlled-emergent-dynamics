"""Shared manifest builders for CLI and experiments.

Notes
-----
Provides deterministic manifest generation without changing existing schema
or values.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def _get_git_commit(cwd: Path) -> str | None:
    """Get current git commit hash from a working directory.

    Parameters
    ----------
    cwd : Path
        Working directory to run git in.

    Returns
    -------
    str | None
        Commit hash or None if not in git repo.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=cwd,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
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
    return _compute_file_hash(spec_path)


def _extract_hypothesis_version(hypothesis_path: Path) -> str:
    """Extract hypothesis version from header or fallback to hash."""
    with open(hypothesis_path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(r"\*\*Version\*\*:\s*(.+)", line.strip())
            if match:
                return match.group(1).strip()
    return _compute_file_hash(hypothesis_path)


def build_experiment_manifest(
    output_dir: Path,
    experiment_name: str,
    seeds: list[int],
    steps: int,
    params: dict[str, Any],
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build the experiment manifest without writing it to disk."""
    result_files: dict[str, str] = {}

    for result_file in output_dir.glob("*.json"):
        if result_file.name != "manifest.json":
            result_files[result_file.name] = _compute_file_hash(result_file)

    resolved_root = repo_root or Path(__file__).resolve().parents[3]
    spec_path = resolved_root / "docs" / "SPEC.md"
    hypothesis_path = resolved_root / "docs" / "HYPOTHESIS.md"

    manifest: dict[str, Any] = {
        "experiment": experiment_name,
        "version": "1.0",
        "git_commit": _get_git_commit(resolved_root),
        "python_version": sys.version,
        "spec_version": _extract_spec_version(spec_path),
        "hypothesis_version": _extract_hypothesis_version(hypothesis_path),
        "spec_path": spec_path.relative_to(resolved_root).as_posix(),
        "hypothesis_path": hypothesis_path.relative_to(resolved_root).as_posix(),
        "seeds": seeds,
        "steps": steps,
        "params": params,
        "result_files": result_files,
    }
    return manifest


def build_sleep_stack_manifest(
    seed: int,
    steps_wake: int,
    steps_sleep: int,
    N: int,
    package_version: str,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build the sleep-stack CLI manifest without writing it to disk."""
    resolved_root = repo_root or Path(__file__).resolve().parents[3]
    manifest: dict[str, Any] = {
        "seed": seed,
        "steps_wake": steps_wake,
        "steps_sleep": steps_sleep,
        "N": N,
        "package_version": package_version,
    }
    sha = _get_git_commit(resolved_root)
    if sha is not None:
        manifest["git_sha"] = sha
    return manifest
