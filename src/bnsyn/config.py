from __future__ import annotations

from pydantic import BaseModel, Field, PositiveFloat


class AdExParams(BaseModel):
    # Units: pF, nS, mV, ms, pA
    C_pF: PositiveFloat = Field(default=150.0, description="Membrane capacitance (pF)")
    gL_nS: PositiveFloat = Field(default=10.0, description="Leak conductance (nS)")
    EL_mV: float = Field(default=-70.0, description="Leak reversal (mV)")
    VT_mV: float = Field(default=-55.0, description="Threshold (mV)")
    DeltaT_mV: PositiveFloat = Field(default=2.0, description="Slope factor (mV)")
    tauw_ms: PositiveFloat = Field(default=200.0, description="Adaptation time constant (ms)")
    a_nS: float = Field(default=2.0, description="Subthreshold adaptation (nS)")
    b_pA: float = Field(default=80.0, description="Spike-triggered adaptation increment (pA)")
    Vreset_mV: float = Field(default=-58.0, description="Reset voltage (mV)")
    Vpeak_mV: float = Field(default=30.0, description="Spike peak clamp (mV)")


class SynapseParams(BaseModel):
    E_AMPA_mV: float = 0.0
    E_NMDA_mV: float = 0.0
    E_GABAA_mV: float = -70.0

    tau_AMPA_ms: PositiveFloat = 2.5
    tau_NMDA_ms: PositiveFloat = 100.0
    tau_GABAA_ms: PositiveFloat = 6.0

    delay_ms: PositiveFloat = 1.0
    mg_mM: PositiveFloat = 1.0  # extracellular Mg2+


class PlasticityParams(BaseModel):
    tau_e_ms: PositiveFloat = 500.0
    tau_plus_ms: PositiveFloat = 20.0
    tau_minus_ms: PositiveFloat = 20.0
    A_plus: PositiveFloat = 1.0
    A_minus: PositiveFloat = 1.05
    eta: PositiveFloat = 5e-3
    w_min: float = 0.0
    w_max: float = 200.0


class CriticalityParams(BaseModel):
    sigma_target: float = 1.0
    eta_sigma: PositiveFloat = 1e-3
    gain_min: PositiveFloat = 0.2
    gain_max: PositiveFloat = 5.0


class TemperatureParams(BaseModel):
    T0: PositiveFloat = 1.0
    Tmin: PositiveFloat = 1e-3
    alpha: float = Field(default=0.95, ge=0.0, le=1.0)
    Tc: PositiveFloat = 0.1
    gate_tau: PositiveFloat = 0.02  # sigmoid sharpness


class DualWeightParams(BaseModel):
    tau_f_s: PositiveFloat = 1800.0  # 30 min
    tau_tag_s: PositiveFloat = 5400.0  # 90 min
    tau_p_s: PositiveFloat = 7200.0  # 2 h
    theta_tag: PositiveFloat = 0.25
    eta_f: PositiveFloat = 0.05
    eta_c: PositiveFloat = 0.005


class EnergyParams(BaseModel):
    lambda_rate: PositiveFloat = 1e-3
    lambda_weight: PositiveFloat = 5e-4
    lambda_energy: PositiveFloat = 1e-2
    r_min_hz: PositiveFloat = 0.05
