# Configuration Reference

This reference summarizes parameter models defined in `src/bnsyn/config.py`.

## Model map

| Model | SPEC linkage |
|---|---|
| `AdExParams` | [SPEC.md](SPEC.md) |
| `SynapseParams` | [SPEC.md](SPEC.md) |
| `PlasticityParams` | [SPEC.md](SPEC.md) |
| `CriticalityParams` | [SPEC.md](SPEC.md) |
| `TemperatureParams` | [SPEC.md](SPEC.md) |
| `DualWeightParams` | [SPEC.md](SPEC.md) |
| `EnergyParams` | [SPEC](SPEC.md) |

## `AdExParams` (units: pF, nS, mV, ms, pA)

| Field | Default |
|---|---|
| `C_pF` | `150.0` |
| `gL_nS` | `10.0` |
| `EL_mV` | `-70.0` |
| `VT_mV` | `-55.0` |
| `DeltaT_mV` | `2.0` |
| `tauw_ms` | `200.0` |
| `a_nS` | `2.0` |
| `b_pA` | `80.0` |
| `Vreset_mV` | `-58.0` |
| `Vpeak_mV` | `30.0` |

## `SynapseParams`

| Field | Default |
|---|---|
| `E_AMPA_mV` | `0.0` |
| `E_NMDA_mV` | `0.0` |
| `E_GABAA_mV` | `-70.0` |
| `tau_AMPA_ms` | `2.5` |
| `tau_NMDA_ms` | `100.0` |
| `tau_GABAA_ms` | `6.0` |
| `delay_ms` | `1.0` |
| `mg_mM` | `1.0` |

## `PlasticityParams`

| Field | Default |
|---|---|
| `tau_e_ms` | `500.0` |
| `tau_plus_ms` | `20.0` |
| `tau_minus_ms` | `20.0` |
| `A_plus` | `1.0` |
| `A_minus` | `1.05` |
| `eta` | `0.005` |
| `w_min` | `0.0` |
| `w_max` | `200.0` |

## `CriticalityParams`

| Field | Default |
|---|---|
| `sigma_target` | `1.0` |
| `eta_sigma` | `0.001` |
| `gain_min` | `0.2` |
| `gain_max` | `5.0` |

## `TemperatureParams`

| Field | Default | Bounds |
|---|---|---|
| `T0` | `1.0` | positive |
| `Tmin` | `0.001` | positive |
| `alpha` | `0.95` | `0.0` to `1.0` |
| `Tc` | `0.1` | positive |
| `gate_tau` | `0.02` | `0.015` to `0.08` |

## `DualWeightParams`

| Field | Default |
|---|---|
| `tau_f_s` | `1800.0` |
| `tau_tag_s` | `5400.0` |
| `tau_p_s` | `7200.0` |
| `theta_tag` | `0.25` |
| `eta_f` | `0.05` |
| `eta_c` | `0.005` |

## `EnergyParams`

| Field | Default |
|---|---|
| `lambda_rate` | `0.001` |
| `lambda_weight` | `0.0005` |
| `lambda_energy` | `0.01` |
| `r_min_hz` | `0.05` |
