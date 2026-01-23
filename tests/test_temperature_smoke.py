from bnsyn.config import TemperatureParams
from bnsyn.temperature.schedule import TemperatureSchedule


def test_temperature_cools_and_gate_changes() -> None:
    sched = TemperatureSchedule(
        TemperatureParams(T0=1.0, Tmin=0.01, alpha=0.9, Tc=0.1, gate_tau=0.02)
    )
    g0 = sched.plasticity_gate()
    for _ in range(20):
        sched.step_geometric()
    g1 = sched.plasticity_gate()
    assert sched.T is not None
    assert sched.T >= 0.01
    assert g1 < g0
