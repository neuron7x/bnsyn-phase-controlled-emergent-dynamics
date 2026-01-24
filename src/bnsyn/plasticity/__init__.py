from .stdp import stdp_kernel
from .three_factor import (
    EligibilityTraces,
    NeuromodulatorTrace,
    three_factor_update,
)

__all__ = ["stdp_kernel", "EligibilityTraces", "NeuromodulatorTrace", "three_factor_update"]
