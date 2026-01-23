from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from bnsyn.production import AdExNeuron, AdExParams, ConnectivityConfig, build_connectivity


pytestmark = pytest.mark.validation


@settings(deadline=None, max_examples=20)
@given(
    n=st.integers(min_value=1, max_value=256),
    dt=st.floats(min_value=1e-5, max_value=5e-3, allow_nan=False, allow_infinity=False),
    current=st.floats(min_value=-5e-9, max_value=5e-9, allow_nan=False, allow_infinity=False),
)
def test_adex_step_finite(n, dt, current):
    neuron = AdExNeuron.init(n=n, params=AdExParams())
    current_vec = np.full((n,), float(current), dtype=np.float64)
    spikes, V = neuron.step(current_vec, float(dt), 0.0)
    assert spikes.shape == (n,)
    assert V.shape == (n,)
    assert np.isfinite(V).all()
    assert np.isfinite(neuron.w).all()


def test_adex_refractory_holds_reset():
    p = AdExParams(t_ref=1e-3, V_spike=-40e-3, V_reset=-60e-3)
    neuron = AdExNeuron.init(n=1, params=p, V0=-45e-3)
    spikes, _ = neuron.step(np.array([5e-9]), 1e-4, 0.0)
    assert bool(spikes[0]) is True

    spikes2, V2 = neuron.step(np.array([5e-9]), 1e-4, 5e-4)
    assert bool(spikes2[0]) is False
    assert abs(float(V2[0]) - p.V_reset) < 1e-12


@settings(deadline=None, max_examples=10)
@given(
    n=st.integers(min_value=2, max_value=512),
    p=st.floats(min_value=0.0, max_value=0.25, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_connectivity_shape_and_diagonal(n, p, seed):
    cfg = ConnectivityConfig(n_pre=n, n_post=n, p_connect=float(p), allow_self=False)
    adj = build_connectivity(cfg, seed=int(seed))
    assert adj.shape == (n, n)
    assert adj.dtype == bool
    assert np.diag(adj).sum() == 0
