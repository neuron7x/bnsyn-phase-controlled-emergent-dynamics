from __future__ import annotations

import importlib


def test_public_imports() -> None:
    modules = [
        "bnsyn",
        "bnsyn.cli",
        "bnsyn.config",
        "bnsyn.rng",
        "bnsyn.sim.network",
        "bnsyn.neuron.adex",
        "bnsyn.synapse.conductance",
        "bnsyn.plasticity.three_factor",
        "bnsyn.criticality.branching",
        "bnsyn.temperature.schedule",
        "bnsyn.connectivity.sparse",
    ]

    for module in modules:
        importlib.import_module(module)
