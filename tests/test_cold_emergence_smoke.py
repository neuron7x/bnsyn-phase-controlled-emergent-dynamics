"""Smoke tests for cold emergence components."""

from __future__ import annotations

import numpy as np
import pytest

from bnsyn.cold_emergence import (
    ColdEmergenceValidator,
    ColdFunctionalSystem,
    ColdPhaseAttractorController,
    IntegratedInformationMetric,
)


@pytest.fixture
def rng() -> np.random.Generator:
    """Create deterministic RNG for tests."""
    return np.random.default_rng(seed=42)


def test_cold_phase_attractor_controller_basic(rng: np.random.Generator) -> None:
    """Test basic attractor controller functionality."""
    controller = ColdPhaseAttractorController(target_sigma=1.0)

    # Create test states
    state = rng.normal(size=10)
    perturbed = state + 1e-8

    # Compute Lyapunov exponent
    lyap = controller.compute_lyapunov_exponent(state, perturbed, dt=0.1)
    assert isinstance(lyap, float)
    assert len(controller.lyapunov_trajectory) == 1

    # Test attractor stabilization
    target = rng.normal(size=10)
    force, temp_reduction = controller.stabilize_cold_attractor(state, target, current_sigma=1.05)
    assert force.shape == state.shape
    assert 0.0 < temp_reduction <= 1.0


def test_cold_phase_is_in_cold_phase() -> None:
    """Test cold phase detection."""
    controller = ColdPhaseAttractorController(target_sigma=1.0)

    # Should be in cold phase with low sigma error and negative Lyapunov
    assert controller.is_in_cold_phase(sigma=1.05, lyapunov=-0.5)
    assert controller.is_in_cold_phase(sigma=0.95, lyapunov=0.05)

    # Should not be in cold phase
    assert not controller.is_in_cold_phase(sigma=1.5, lyapunov=0.5)
    assert not controller.is_in_cold_phase(sigma=0.5, lyapunov=-0.5)


def test_integrated_information_metric_basic(rng: np.random.Generator) -> None:
    """Test Φ computation."""
    metric = IntegratedInformationMetric()

    state = rng.random(size=20)
    phi = metric.compute_phi(state, partition_size=5)

    assert isinstance(phi, float)
    assert phi >= 0.0


def test_integrated_information_metric_synergy(rng: np.random.Generator) -> None:
    """Test synergy computation."""
    metric = IntegratedInformationMetric()

    state = rng.random(size=20)
    synergy = metric.compute_synergy(state)

    assert isinstance(synergy, float)
    assert 0.0 <= synergy <= 1.0


def test_integrated_information_metric_edge_cases() -> None:
    """Test edge cases for information metrics."""
    metric = IntegratedInformationMetric()

    # Empty state
    phi_empty = metric.compute_phi(np.array([]))
    assert phi_empty == 0.0

    # Small state
    phi_small = metric.compute_phi(np.array([1.0, 2.0]))
    assert phi_small >= 0.0

    # Synergy with small state
    synergy = metric.compute_synergy(np.array([1.0]))
    assert synergy == 0.5  # Default for degenerate case


def test_cold_functional_system_basic(rng: np.random.Generator) -> None:
    """Test functional system without affective modulation."""
    goal = rng.normal(size=10)
    system = ColdFunctionalSystem(goal_representation=goal, error_threshold=0.1)

    # Test afferent synthesis
    sensory = rng.normal(size=5)
    memory = rng.normal(size=5)
    integrated = system.afferent_synthesis(sensory, memory)

    assert integrated.shape == (10,)
    assert system.afferent_synthesis_state is not None


def test_cold_functional_system_acceptor(rng: np.random.Generator) -> None:
    """Test acceptor of result mechanism."""
    goal = np.zeros(10)
    system = ColdFunctionalSystem(goal_representation=goal)

    # Test with close prediction
    close_pred = np.zeros(10) + 0.05
    error_close = system.acceptor_of_result(close_pred)
    assert error_close < 1.0

    # Test with far prediction
    far_pred = np.ones(10) * 5.0
    error_far = system.acceptor_of_result(far_pred)
    assert error_far > error_close


def test_cold_functional_system_execute(rng: np.random.Generator) -> None:
    """Test cold program execution."""
    goal = rng.normal(size=10)
    system = ColdFunctionalSystem(goal_representation=goal, error_threshold=0.5)

    action = rng.normal(size=5)
    sensory = rng.normal(size=5)
    memory = rng.normal(size=5)

    success, correction = system.execute_cold_program(action, sensory, memory)

    assert isinstance(success, bool)
    assert isinstance(correction, float)
    assert correction >= 0.0


def test_cold_functional_system_goal_update(rng: np.random.Generator) -> None:
    """Test goal update mechanism."""
    goal = rng.normal(size=10)
    system = ColdFunctionalSystem(goal_representation=goal)

    new_goal = rng.normal(size=10)
    system.update_goal(new_goal)
    assert np.array_equal(system.goal_representation, new_goal)

    # Test shape mismatch
    with pytest.raises(ValueError, match="Goal shape mismatch"):
        system.update_goal(rng.normal(size=5))


def test_cold_emergence_validator_basic(rng: np.random.Generator) -> None:
    """Test cold emergence validation."""
    validator = ColdEmergenceValidator()

    state = rng.random(size=20)
    lyapunov = -0.2  # Deterministic
    temperature = 0.05  # Cold

    result = validator.validate_cold_emergence(state, lyapunov, temperature)

    assert "lyapunov_exponent" in result
    assert "integrated_information_phi" in result
    assert "synergy_vs_redundancy" in result
    assert "is_truly_cold_emergent" in result
    assert isinstance(result["is_truly_cold_emergent"], bool)


def test_cold_emergence_validator_determinism() -> None:
    """Test determinism criterion."""
    validator = ColdEmergenceValidator(lyapunov_threshold=-0.1)

    state = np.random.random(20)

    # Deterministic system
    result_det = validator.validate_cold_emergence(state, lyapunov_exponent=-0.5, temperature=0.1)
    assert result_det["is_deterministic"]

    # Non-deterministic system
    result_nondet = validator.validate_cold_emergence(state, lyapunov_exponent=0.5, temperature=0.1)
    assert not result_nondet["is_deterministic"]


def test_cold_emergence_validator_trajectory(rng: np.random.Generator) -> None:
    """Test Lyapunov measurement from trajectory."""
    validator = ColdEmergenceValidator()

    # Create damped trajectory (deterministic)
    trajectory = [rng.normal(size=10) * (0.9**i) for i in range(20)]

    lyap = validator.measure_lyapunov(trajectory, dt=0.1)
    assert isinstance(lyap, float)


def test_cold_emergence_validator_integration(rng: np.random.Generator) -> None:
    """Test integration measurement."""
    validator = ColdEmergenceValidator()

    state = rng.random(size=20)
    phi = validator.measure_integrated_info(state)

    assert isinstance(phi, float)
    assert phi >= 0.0


def test_cold_emergence_validator_synergy(rng: np.random.Generator) -> None:
    """Test synergy measurement."""
    validator = ColdEmergenceValidator()

    state = rng.random(size=20)
    synergy = validator.measure_synergy(state)

    assert isinstance(synergy, float)
    assert 0.0 <= synergy <= 1.0
