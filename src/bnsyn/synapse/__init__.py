"""Synapse models for BN-Syn conductance dynamics."""

from .conductance import (
    ConductanceState as ConductanceState,
    ConductanceSynapses as ConductanceSynapses,
)

__all__ = ["ConductanceState", "ConductanceSynapses"]
