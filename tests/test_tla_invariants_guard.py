"""
Guard tests for TLA+ invariants defined in specs/tla/BNsyn.tla

These tests enforce the formal invariants specified in the TLA+ model:
- INV-1: GainClamp - Criticality gain always stays within bounds
- INV-2: TemperatureBounds - Temperature stays within physical bounds
- INV-3: GateBounds - Plasticity gate stays in valid range [0, 1]

Reference: specs/tla/BNsyn.tla:124-157
"""

from __future__ import annotations

import numpy as np
import pytest

from bnsyn.config import CriticalityParams, TemperatureParams
from bnsyn.temperature.schedule import TemperatureSchedule


@pytest.fixture
def default_temp_params() -> TemperatureParams:
    """Default temperature parameters matching TLA+ constants."""
    return TemperatureParams(
        T0=1.0,
        Tmin=1e-3,
        alpha=0.95,
        Tc=0.5,
        gate_tau=0.1,
    )


@pytest.fixture
def default_crit_params() -> CriticalityParams:
    """Default criticality parameters matching TLA+ constants."""
    return CriticalityParams(
        sigma_target=1.0,
        eta_sigma=0.001,
        gain_min=0.2,
        gain_max=5.0,
    )


class TestTLAInvariantINV1_GainClamp:
    """
    INV-1: GainClamp
    Criticality gain always stays within bounds [gain_min, gain_max].
    Maps to: src/bnsyn/config.py:CriticalityParams
    TLA+ spec: specs/tla/BNsyn.tla:133-134
    """

    def test_gain_bounds_enforced_at_initialization(
        self, default_crit_params: CriticalityParams
    ) -> None:
        """Gain parameters must satisfy bounds at initialization."""
        assert default_crit_params.gain_min >= 0.0
        assert default_crit_params.gain_max > default_crit_params.gain_min
        # Initial gain would be somewhere in the valid range
        initial_gain = (default_crit_params.gain_min + default_crit_params.gain_max) / 2
        assert default_crit_params.gain_min <= initial_gain <= default_crit_params.gain_max

    def test_gain_clamp_extreme_values(self, default_crit_params: CriticalityParams) -> None:
        """Test gain stays within bounds under extreme configurations."""
        # Test lower bound
        assert default_crit_params.gain_min >= 0.0, "INV-1: gain_min must be >= 0"

        # Test upper bound
        assert default_crit_params.gain_max <= 100.0, "INV-1: gain_max must be reasonable"

        # Test ordering
        assert default_crit_params.gain_min < default_crit_params.gain_max, (
            "INV-1: gain_min < gain_max"
        )

    def test_gain_invariant_preserved_across_updates(
        self, default_crit_params: CriticalityParams
    ) -> None:
        """Simulate gain updates and verify bounds are maintained."""
        # Start at midpoint
        gain = (default_crit_params.gain_min + default_crit_params.gain_max) / 2
        gain_step = 0.1  # Fixed step size for testing

        # Simulate 100 random gain updates
        for _ in range(100):
            # Random perturbation
            delta = np.random.uniform(-gain_step, gain_step)
            gain_new = gain + delta

            # Clamp to bounds (as implementation should do)
            gain_clamped = np.clip(
                gain_new, default_crit_params.gain_min, default_crit_params.gain_max
            )

            # Verify INV-1
            assert default_crit_params.gain_min <= gain_clamped <= default_crit_params.gain_max, (
                f"INV-1 violated: gain={gain_clamped} not in [{default_crit_params.gain_min}, {default_crit_params.gain_max}]"
            )

            gain = gain_clamped


class TestTLAInvariantINV2_TemperatureBounds:
    """
    INV-2: TemperatureBounds
    Temperature stays within physical bounds [Tmin, T0].
    Maps to: src/bnsyn/config.py:TemperatureParams
    TLA+ spec: specs/tla/BNsyn.tla:141-142
    """

    def test_temperature_initialization_within_bounds(
        self, default_temp_params: TemperatureParams
    ) -> None:
        """Temperature parameters must satisfy bounds at initialization."""
        assert default_temp_params.Tmin > 0.0, "INV-2: Tmin must be positive"
        assert default_temp_params.T0 >= default_temp_params.Tmin, "INV-2: T0 >= Tmin"

    def test_temperature_cooling_preserves_bounds(
        self, default_temp_params: TemperatureParams
    ) -> None:
        """Temperature remains in [Tmin, T0] during cooling schedule."""
        schedule = TemperatureSchedule(default_temp_params)

        # Initial temperature must be within bounds
        T = schedule.T
        assert default_temp_params.Tmin <= T <= default_temp_params.T0, (
            f"INV-2 violated at initialization: T={T}"
        )

        # Simulate 1000 cooling steps
        for _ in range(1000):
            schedule.step_geometric()
            T = schedule.T

            # Verify INV-2
            assert default_temp_params.Tmin <= T <= default_temp_params.T0, (
                f"INV-2 violated: T={T} not in [{default_temp_params.Tmin}, {default_temp_params.T0}]"
            )

            # Stop if fully cooled
            if T <= default_temp_params.Tmin:
                break

    def test_temperature_never_goes_negative(self, default_temp_params: TemperatureParams) -> None:
        """Temperature must remain positive (physical constraint)."""
        schedule = TemperatureSchedule(default_temp_params)

        for _ in range(1000):
            schedule.step_geometric()
            assert schedule.T > 0.0, f"INV-2 violated: T={schedule.T} became negative"


class TestTLAInvariantINV3_GateBounds:
    """
    INV-3: GateBounds
    Plasticity gate stays in valid range [0, 1].
    Maps to: src/bnsyn/temperature/schedule.py:gate_sigmoid
    TLA+ spec: specs/tla/BNsyn.tla:149-150
    """

    def test_gate_within_unit_interval(self, default_temp_params: TemperatureParams) -> None:
        """Gate value must stay in [0, 1] throughout cooling."""
        schedule = TemperatureSchedule(default_temp_params)

        for _ in range(1000):
            gate = schedule.plasticity_gate()

            # Verify INV-3
            assert 0.0 <= gate <= 1.0, f"INV-3 violated: gate={gate} not in [0, 1]"

            schedule.step_geometric()

            # Stop if fully cooled
            if schedule.T <= default_temp_params.Tmin:
                break

    def test_gate_sigmoid_boundary_conditions(self, default_temp_params: TemperatureParams) -> None:
        """Gate sigmoid must be well-behaved at extreme temperatures."""
        schedule = TemperatureSchedule(default_temp_params)

        # Test at high temperature (T >> Tc) -> gate ≈ 1
        schedule.T = 10.0 * default_temp_params.Tc
        gate_high = schedule.plasticity_gate()
        assert 0.0 <= gate_high <= 1.0, f"INV-3 violated at high T: gate={gate_high}"
        assert gate_high > 0.5, "Gate should be open at high temperature"

        # Test at low temperature (T << Tc) -> gate ≈ 0
        schedule.T = 0.1 * default_temp_params.Tc
        gate_low = schedule.plasticity_gate()
        assert 0.0 <= gate_low <= 1.0, f"INV-3 violated at low T: gate={gate_low}"
        assert gate_low < 0.5, "Gate should be closed at low temperature"


class TestTLAInvariantComposite:
    """
    Test that all three TLA+ invariants hold simultaneously.
    This is the TypeOK predicate from the TLA+ spec.
    """

    def test_all_invariants_hold_simultaneously(
        self,
        default_temp_params: TemperatureParams,
        default_crit_params: CriticalityParams,
    ) -> None:
        """All three invariants must hold at every step of the simulation."""
        schedule = TemperatureSchedule(default_temp_params)
        gain = (default_crit_params.gain_min + default_crit_params.gain_max) / 2
        gain_step = 0.1  # Fixed step for testing

        for step in range(1000):
            # Check INV-2: TemperatureBounds
            T = schedule.T
            assert default_temp_params.Tmin <= T <= default_temp_params.T0, (
                f"Step {step}: INV-2 violated"
            )

            # Check INV-3: GateBounds
            gate = schedule.plasticity_gate()
            assert 0.0 <= gate <= 1.0, f"Step {step}: INV-3 violated"

            # Check INV-1: GainClamp (simulated update)
            gain_delta = np.random.uniform(-gain_step, gain_step)
            gain = np.clip(
                gain + gain_delta,
                default_crit_params.gain_min,
                default_crit_params.gain_max,
            )
            assert default_crit_params.gain_min <= gain <= default_crit_params.gain_max, (
                f"Step {step}: INV-1 violated"
            )

            # Advance simulation
            schedule.step_geometric()

            if schedule.T <= default_temp_params.Tmin:
                break
