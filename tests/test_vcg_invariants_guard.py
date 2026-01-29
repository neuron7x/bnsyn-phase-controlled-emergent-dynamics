import numpy as np

from bnsyn.vcg import VCGParams, allocation_multiplier, update_support_level, update_support_vector


def test_vcg_update_support_level_is_deterministic() -> None:
    params = VCGParams(theta_c=1.0, alpha_down=0.2, alpha_up=0.1, epsilon=0.05)

    cases = [(0.2, 0.8), (1.2, 0.4), (0.0, 0.0), (2.0, 1.0)]
    for contribution, support in cases:
        first = update_support_level(contribution, support, params)
        second = update_support_level(contribution, support, params)
        assert first == second
        assert 0.0 <= first <= 1.0


def test_vcg_allocation_multiplier_is_pure_and_bounded() -> None:
    params = VCGParams(theta_c=1.0, alpha_down=0.2, alpha_up=0.1, epsilon=0.05)

    supports = [0.0, 0.5, 1.0]
    for support in supports:
        first = allocation_multiplier(support, params)
        second = allocation_multiplier(support, params)
        assert first == second
        assert params.epsilon <= first <= 1.0


def test_vcg_update_support_vector_is_pure() -> None:
    params = VCGParams(theta_c=1.0, alpha_down=0.2, alpha_up=0.1, epsilon=0.05)

    contributions = np.array([0.0, 2.0, 0.5], dtype=float)
    support = np.array([0.8, 0.1, 0.0], dtype=float)
    contributions_copy = contributions.copy()
    support_copy = support.copy()

    updated = update_support_vector(contributions, support, params)

    np.testing.assert_array_equal(contributions, contributions_copy)
    np.testing.assert_array_equal(support, support_copy)
    assert updated.shape == support.shape
    assert np.all((0.0 <= updated) & (updated <= 1.0))
