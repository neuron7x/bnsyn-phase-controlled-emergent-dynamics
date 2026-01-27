"""Pydantic models for experiment configuration schema.

Parameters
----------
None

Returns
-------
None

Notes
-----
Auto-generated from schemas/experiment.schema.json.
Provides type-safe experiment configuration with validation.

References
----------
docs/LEGENDARY_QUICKSTART.md
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ExperimentConfig(BaseModel):
    """Experiment metadata configuration.

    Parameters
    ----------
    name : str
        Experiment name (lowercase alphanumeric, underscore, dash)
    version : str
        Version identifier (e.g., v1, v2)
    seeds : list[int]
        Random seeds for reproducibility (1-100 unique positive integers)
    """

    name: str = Field(..., pattern=r"^[a-z0-9_-]+$")
    version: str = Field(..., pattern=r"^v[0-9]+$")
    seeds: list[int] = Field(..., min_length=1, max_length=100)

    model_config = {"extra": "forbid"}


class NetworkConfig(BaseModel):
    """Network configuration.

    Parameters
    ----------
    size : int
        Number of neurons (10-100000)
    """

    size: int = Field(..., ge=10, le=100000)

    model_config = {"extra": "forbid"}


class SimulationConfig(BaseModel):
    """Simulation parameters configuration.

    Parameters
    ----------
    duration_ms : float
        Simulation duration in milliseconds (≥1)
    dt_ms : Literal[0.01, 0.05, 0.1, 0.5, 1.0]
        Timestep in milliseconds
    """

    duration_ms: float = Field(..., ge=1)
    dt_ms: Literal[0.01, 0.05, 0.1, 0.5, 1.0]

    model_config = {"extra": "forbid"}


class BNSynExperimentConfig(BaseModel):
    """Complete BN-Syn experiment configuration.

    Parameters
    ----------
    experiment : ExperimentConfig
        Experiment metadata
    network : NetworkConfig
        Network configuration
    simulation : SimulationConfig
        Simulation parameters

    Examples
    --------
    Load from YAML::

        import yaml
        from bnsyn.schemas.experiment import BNSynExperimentConfig

        with open("config.yaml") as f:
            data = yaml.safe_load(f)
        config = BNSynExperimentConfig(**data)
    """

    experiment: ExperimentConfig
    network: NetworkConfig
    simulation: SimulationConfig

    model_config = {"extra": "forbid"}
