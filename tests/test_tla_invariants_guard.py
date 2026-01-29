from bnsyn.config import CriticalityParams, TemperatureParams
from bnsyn.criticality.branching import SigmaController
from bnsyn.temperature.schedule import TemperatureSchedule


def test_inv1_gain_clamp_uses_sigma_controller() -> None:
    params = CriticalityParams(sigma_target=1.0, eta_sigma=10.0, gain_min=0.2, gain_max=2.0)
    controller = SigmaController(params=params, gain=1.0)

    sigmas = [10.0, 0.0, 10.0, 0.0]
    gains: list[float] = []
    for sigma in sigmas:
        gain = controller.step(sigma)
        gains.append(gain)
        assert params.gain_min <= gain <= params.gain_max

    assert min(gains) == params.gain_min
    assert max(gains) == params.gain_max


def test_inv2_temperature_bounds_use_schedule_step() -> None:
    params = TemperatureParams(T0=1.0, Tmin=1e-3, alpha=0.5, Tc=0.1, gate_tau=0.02)
    schedule = TemperatureSchedule(params=params)

    temperatures = [schedule.step_geometric() for _ in range(8)]
    for temperature in temperatures:
        assert params.Tmin <= temperature <= params.T0


def test_inv3_gate_bounds_use_schedule_gate() -> None:
    params = TemperatureParams(T0=1.0, Tmin=1e-3, alpha=0.5, Tc=0.1, gate_tau=0.02)
    schedule = TemperatureSchedule(params=params)

    gate_values = [schedule.plasticity_gate()]
    for _ in range(6):
        schedule.step_geometric()
        gate_values.append(schedule.plasticity_gate())

    for gate in gate_values:
        assert 0.0 <= gate <= 1.0
