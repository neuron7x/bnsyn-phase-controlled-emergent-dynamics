from .stdp import stdp_kernel as stdp_kernel
from .three_factor import (
    EligibilityTraces as EligibilityTraces,
    NeuromodulatorTrace as NeuromodulatorTrace,
    three_factor_update as three_factor_update,
)

__all__ = ["stdp_kernel", "EligibilityTraces", "NeuromodulatorTrace", "three_factor_update"]
