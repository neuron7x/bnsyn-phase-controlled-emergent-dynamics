"""Pydantic models for experiment configuration schema.

Auto-generated from schemas/experiment.schema.json.
Provides type-safe experiment configuration with validation.

References
----------
docs/LEGENDARY_QUICKSTART.md
"""

from __future__ import annotations

from math import isclose

from pydantic import BaseModel, Field, field_validator, model_validator


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

    @field_validator("seeds")
    @classmethod
    def validate_seeds(cls, v: list[int]) -> list[int]:
        """Validate that seeds are unique positive integers."""
        if any(not isinstance(seed, int) or isinstance(seed, bool) or seed <= 0 for seed in v):
            raise ValueError("seeds must contain only positive integers")
        if len(set(v)) != len(v):
            raise ValueError("seeds must be unique positive integers")
        return v

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
    dt_ms : float
        Timestep in milliseconds (must be 0.01, 0.05, 0.1, 0.5, or 1.0)
    """

    duration_ms: float = Field(..., ge=1)
    dt_ms: float

    @field_validator("dt_ms")
    @classmethod
    def validate_dt_ms(cls, v: float) -> float:
        """Validate that dt_ms is one of the allowed values."""
        allowed_values = [0.01, 0.05, 0.1, 0.5, 1.0]
        if v not in allowed_values:
            raise ValueError(f"dt_ms must be one of {allowed_values}, got {v}")
        return v

    @model_validator(mode="after")
    def validate_duration_multiple(self) -> "SimulationConfig":
        """Validate that duration_ms is an integer multiple of dt_ms."""
        ratio = self.duration_ms / self.dt_ms
        nearest = round(ratio)
        if not isclose(ratio, nearest, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                "duration_ms must be an integer multiple of dt_ms within tolerance; "
                f"got duration_ms={self.duration_ms}, dt_ms={self.dt_ms}"
            )
        return self

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
