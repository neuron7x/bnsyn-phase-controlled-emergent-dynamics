import numpy as np
import pytest

from bnsyn.criticality.analysis import fit_power_law_mle, mr_branching_ratio


@pytest.mark.validation
def test_mr_branching_ratio_geometric_decay() -> None:
    activity = np.array([1.0, 0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144])
    sigma = mr_branching_ratio(activity, max_lag=3)
    assert sigma == pytest.approx(0.8, abs=1e-3)


@pytest.mark.validation
def test_power_law_mle_matches_formula() -> None:
    data = np.array([1.0, 2.0, 4.0, 8.0])
    fit = fit_power_law_mle(data, xmin=1.0)
    expected = 1.0 + len(data) / float(np.sum(np.log(data / 1.0)))
    assert fit.alpha == pytest.approx(expected)
    assert fit.xmin == pytest.approx(1.0)
