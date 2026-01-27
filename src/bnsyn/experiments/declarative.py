"""Declarative experiment execution from YAML configurations.

Provides YAML-driven experiment runner with schema validation.

References
----------
docs/LEGENDARY_QUICKSTART.md
schemas/experiment.schema.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from bnsyn.schemas.experiment import BNSynExperimentConfig
from bnsyn.sim.network import run_simulation


def load_config(config_path: str | Path) -> BNSynExperimentConfig:
    """Load and validate experiment configuration from YAML file.

    Parameters
    ----------
    config_path : str | Path
        Path to YAML configuration file

    Returns
    -------
    BNSynExperimentConfig
        Validated configuration object

    Raises
    ------
    FileNotFoundError
        If config file doesn't exist
    ValueError
        If YAML is invalid or doesn't match schema

    Examples
    --------
    Load configuration::

        from bnsyn.experiments.declarative import load_config

        config = load_config("examples/configs/quickstart.yaml")
        print(f"Running {config.experiment.name} v{config.experiment.version}")
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {config_path}: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML object, got {type(data).__name__}")

    try:
        return BNSynExperimentConfig(**data)
    except Exception as e:
        # Provide helpful error message
        msg = f"❌ Config validation failed: {config_path}\n\nError: {e}"
        raise ValueError(msg) from e


def run_experiment(config: BNSynExperimentConfig) -> dict[str, Any]:
    """Run experiment from validated configuration.

    Parameters
    ----------
    config : BNSynExperimentConfig
        Validated experiment configuration

    Returns
    -------
    dict[str, Any]
        Experiment results with metrics for each seed

    Notes
    -----
    Executes simulation for each seed and aggregates results.

    Examples
    --------
    Run experiment::

        from bnsyn.experiments.declarative import load_config, run_experiment

        config = load_config("examples/configs/quickstart.yaml")
        results = run_experiment(config)
        print(f"Completed {len(results['runs'])} runs")
    """
    results: dict[str, Any] = {
        "config": {
            "name": config.experiment.name,
            "version": config.experiment.version,
            "network_size": config.network.size,
            "duration_ms": config.simulation.duration_ms,
            "dt_ms": config.simulation.dt_ms,
        },
        "runs": [],
    }

    steps = int(config.simulation.duration_ms / config.simulation.dt_ms)

    for seed in config.experiment.seeds:
        metrics = run_simulation(
            steps=steps, dt_ms=config.simulation.dt_ms, seed=seed, N=config.network.size
        )
        results["runs"].append({"seed": seed, "metrics": metrics})

    return results


def run_from_yaml(config_path: str | Path, output_path: str | Path | None = None) -> None:
    """Load config from YAML, run experiment, and save results.

    Parameters
    ----------
    config_path : str | Path
        Path to YAML configuration file
    output_path : str | Path | None, optional
        Path to save results JSON (default: None = print to stdout)

    Returns
    -------
    None

    Examples
    --------
    Run and save results::

        from bnsyn.experiments.declarative import run_from_yaml

        run_from_yaml(
            "examples/configs/quickstart.yaml",
            "results/quickstart_v1.json"
        )
    """
    config = load_config(config_path)
    print(f"✓ Config validated: {config.experiment.name} {config.experiment.version}")
    print(
        f"  Network: N={config.network.size}, "
        f"Duration: {config.simulation.duration_ms}ms, "
        f"dt: {config.simulation.dt_ms}ms"
    )
    print(f"  Seeds: {len(config.experiment.seeds)} runs")

    results = run_experiment(config)

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(results, f, indent=2, sort_keys=True)
        print(f"✓ Results saved to {output_path}")
    else:
        print(json.dumps(results, indent=2, sort_keys=True))
