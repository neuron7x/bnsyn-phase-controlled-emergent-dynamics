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

from bnsyn.experiments.emergence import run_emergence_to_disk
from bnsyn.numerics import compute_steps_exact
from bnsyn.schemas.experiment import BNSynExperimentConfig
from bnsyn.sim.network import run_simulation


def load_config(config_path: str | Path) -> BNSynExperimentConfig:
    """Load and validate experiment configuration from YAML file."""
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
        msg = f"❌ Config validation failed: {config_path}\n\nError: {e}"
        raise ValueError(msg) from e


def run_experiment(config: BNSynExperimentConfig) -> dict[str, Any]:
    """Run experiment from validated configuration."""
    results: dict[str, Any] = {
        "config": {
            "name": config.experiment.name,
            "version": config.experiment.version,
            "network_size": config.network.size,
            "duration_ms": config.simulation.duration_ms,
            "dt_ms": config.simulation.dt_ms,
            "external_current_pA": config.simulation.external_current_pA,
        },
        "runs": [],
    }

    steps = compute_steps_exact(config.simulation.duration_ms, config.simulation.dt_ms)

    for seed in config.experiment.seeds:
        if config.simulation.artifact_dir is None:
            metrics = run_simulation(
                steps=steps,
                dt_ms=config.simulation.dt_ms,
                seed=seed,
                N=config.network.size,
                external_current_pA=config.simulation.external_current_pA,
            )
            results["runs"].append({"seed": seed, "metrics": metrics})
        else:
            metrics, artifact_npz = run_emergence_to_disk(
                N=config.network.size,
                dt_ms=config.simulation.dt_ms,
                duration_ms=config.simulation.duration_ms,
                seed=seed,
                external_current_pA=config.simulation.external_current_pA,
                output_dir=config.simulation.artifact_dir,
            )
            results["runs"].append({"seed": seed, "metrics": metrics, "artifact_npz": artifact_npz})

    return results


def run_from_yaml(config_path: str | Path, output_path: str | Path | None = None) -> None:
    """Load config from YAML, run experiment, and save results."""
    config = load_config(config_path)
    print(f"✓ Config validated: {config.experiment.name} {config.experiment.version}")
    print(
        f"  Network: N={config.network.size}, "
        f"Duration: {config.simulation.duration_ms}ms, "
        f"dt: {config.simulation.dt_ms}ms"
    )
    print(f"  external_current_pA: {config.simulation.external_current_pA}")
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
