from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bnsyn.config import TemperatureParams


def gate_sigmoid(T: float, Tc: float, tau: float) -> float:
    # G(T) = 1/(1+exp(-(T-Tc)/tau))
    return float(1.0 / (1.0 + np.exp(-(float(T) - float(Tc)) / float(tau))))


@dataclass
class TemperatureSchedule:
    params: TemperatureParams
    T: float | None = None

    def __post_init__(self) -> None:
        if self.T is None:
            self.T = float(self.params.T0)

    def step_geometric(self) -> float:
        p = self.params
        temperature = self.T if self.T is not None else float(p.T0)
        self.T = max(float(p.Tmin), float(temperature) * float(p.alpha))
        return float(self.T)

    def plasticity_gate(self) -> float:
        p = self.params
        temperature = self.T if self.T is not None else float(p.T0)
        return gate_sigmoid(float(temperature), float(p.Tc), float(p.gate_tau))
