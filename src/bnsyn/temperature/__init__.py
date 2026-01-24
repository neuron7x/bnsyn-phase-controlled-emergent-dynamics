"""Temperature schedules and gating functions."""

from .schedule import TemperatureSchedule as TemperatureSchedule, gate_sigmoid as gate_sigmoid

__all__ = ["TemperatureSchedule", "gate_sigmoid"]
