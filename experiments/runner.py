"""Experiment runner CLI.

This module provides a command-line interface for running BN-Syn experiments
and generating reproducible results with manifests.

Usage
-----
python -m experiments.runner temp_ablation_v1
python -m experiments.runner temp_ablation_v1 --seeds 5 --out results/_smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from experiments.registry import get_experiment_config
from experiments.temperature_ablation_consolidation import run_temperature_ablation_experiment


def get_git_commit() -> str | None:
    """Get current git commit hash.

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
            cwd=Path(__file__).parent.parent,
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


def generate_manifest(
    output_dir: Path,
    experiment_name: str,
    seeds: list[int],
    steps: int,
    params: dict[str, Any],
) -> None:
    """Generate reproducibility manifest.

    Parameters
    ----------
    output_dir : Path
        Output directory containing results.
    experiment_name : str
        Experiment identifier.
    seeds : list[int]
        Random seeds used.
    steps : int
        Number of steps per trial.
    params : dict[str, Any]
        Experiment parameters.
    """
    manifest = {
        "experiment": experiment_name,
        "version": "1.0",
        "git_commit": get_git_commit(),
        "python_version": sys.version,
        "seeds": seeds,
        "steps": steps,
        "params": params,
        "result_files": {},
    }

    # Compute hashes for all result files
    for result_file in output_dir.glob("*.json"):
        if result_file.name != "manifest.json":
            manifest["result_files"][result_file.name] = compute_file_hash(result_file)

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest saved to {manifest_path}")


def run_experiment(experiment_name: str, seeds: int | None, output_dir: str | None) -> int:
    """Run an experiment and generate results.

    Parameters
    ----------
    experiment_name : str
        Experiment identifier.
    seeds : int | None
        Number of seeds (None = use default).
    output_dir : str | None
        Output directory (None = use default).

    Returns
    -------
    int
        Exit code (0 = success, 1 = failure).
    """
    try:
        config = get_experiment_config(experiment_name)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Determine seeds and output directory
    num_seeds = seeds if seeds is not None else config.default_seeds
    seed_list = list(range(num_seeds))

    if output_dir is None:
        output_path = Path("results") / experiment_name
    else:
        output_path = Path(output_dir)

    print(f"Running experiment: {config.name}")
    print(f"Description: {config.description}")
    print(f"Seeds: {num_seeds}")
    print(f"Steps: {config.default_steps}")
    print(f"Output: {output_path}")

    # Run experiment
    if experiment_name == "temp_ablation_v1":
        run_temperature_ablation_experiment(
            seeds=seed_list,
            steps=config.default_steps,
            output_dir=output_path,
            params=config.params,
        )
    else:
        print(f"Error: Unknown experiment implementation: {experiment_name}", file=sys.stderr)
        return 1

    # Generate manifest
    generate_manifest(
        output_dir=output_path,
        experiment_name=experiment_name,
        seeds=seed_list,
        steps=config.default_steps,
        params=config.params,
    )

    print(f"\nExperiment complete: {experiment_name}")
    print(f"Results: {output_path}")
    return 0


def main() -> int:
    """Main CLI entry point.

    Returns
    -------
    int
        Exit code.
    """
    parser = argparse.ArgumentParser(
        description="Run BN-Syn experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "experiment",
        type=str,
        help="Experiment name (e.g., temp_ablation_v1)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Number of seeds (default: experiment default)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory (default: results/<experiment>)",
    )

    args = parser.parse_args()
    return run_experiment(args.experiment, args.seeds, args.out)


if __name__ == "__main__":
    sys.exit(main())
