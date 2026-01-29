import numpy as np
import pytest

from bnsyn.vcg import VCGParams, allocation_multiplier, update_support_level, update_support_vector


def test_vcg_support_updates() -> None:
    params = VCGParams(theta_c=1.0, alpha_down=0.2, alpha_up=0.1, epsilon=0.05)
    support = update_support_level(contribution=0.5, support=1.0, params=params)
    assert support == 0.8
    support = update_support_level(contribution=1.5, support=support, params=params)
    assert support == 0.9
    multiplier = allocation_multiplier(support, params)
    assert 0.05 <= multiplier <= 1.0


def test_vcg_vector_update() -> None:
    params = VCGParams(theta_c=1.0, alpha_down=0.3, alpha_up=0.2, epsilon=0.1)
    contributions = np.array([0.2, 1.2])
    support = np.array([0.9, 0.4])
    updated = update_support_vector(contributions, support, params)
    assert updated[0] == pytest.approx(0.6)
    assert updated[1] == pytest.approx(0.6)


def test_vcg_param_validation() -> None:
    with pytest.raises(ValueError, match="theta_c must be non-negative"):
        VCGParams(theta_c=-0.1, alpha_down=0.1, alpha_up=0.1, epsilon=0.5)
    with pytest.raises(ValueError, match="alpha_down and alpha_up must be non-negative"):
        VCGParams(theta_c=0.1, alpha_down=-0.1, alpha_up=0.1, epsilon=0.5)
    with pytest.raises(ValueError, match="alpha_down and alpha_up must be non-negative"):
        VCGParams(theta_c=0.1, alpha_down=0.1, alpha_up=-0.1, epsilon=0.5)
    with pytest.raises(ValueError, match="epsilon must be in \\[0, 1\\]"):
        VCGParams(theta_c=0.1, alpha_down=0.1, alpha_up=0.1, epsilon=1.1)


def test_vcg_input_validation() -> None:
    params = VCGParams(theta_c=1.0, alpha_down=0.2, alpha_up=0.1, epsilon=0.05)
    with pytest.raises(ValueError, match="support must be in \\[0, 1\\]"):
        update_support_level(contribution=0.5, support=1.2, params=params)
    with pytest.raises(ValueError, match="support must be in \\[0, 1\\]"):
        allocation_multiplier(support=-0.1, params=params)
    with pytest.raises(ValueError, match="contributions and support must have the same shape"):
        update_support_vector(np.array([0.5, 0.2]), np.array([0.1]), params)
