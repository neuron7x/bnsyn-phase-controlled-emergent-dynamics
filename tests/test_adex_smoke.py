import numpy as np
import pytest

from bnsyn.config import AdExParams
from bnsyn.neuron.adex import AdExState, adex_step, adex_step_adaptive, adex_step_with_error_tracking


def test_adex_step_spikes_and_resets() -> None:
    p = AdExParams()
    V = np.array([p.Vpeak_mV + 1.0, p.EL_mV], dtype=float)
    w = np.zeros_like(V)
    s = AdExState(V_mV=V, w_pA=w, spiked=np.zeros_like(V, dtype=bool))
    out = adex_step(s, p, dt_ms=0.1, I_syn_pA=np.zeros(2), I_ext_pA=np.zeros(2))
    assert out.spiked[0]
    assert out.V_mV[0] == p.Vreset_mV
    assert out.w_pA[0] > 0


def test_adex_step_with_error_tracking_reports_metrics() -> None:
    p = AdExParams()
    V = np.array([p.EL_mV - 5.0, p.EL_mV + 2.0], dtype=float)
    w = np.zeros_like(V)
    state = AdExState(V_mV=V, w_pA=w, spiked=np.zeros_like(V, dtype=bool))
    out, metrics = adex_step_with_error_tracking(
        state,
        p,
        dt_ms=0.1,
        I_syn_pA=np.zeros(2),
        I_ext_pA=np.zeros(2),
    )
    assert out.V_mV.shape == V.shape
    assert out.w_pA.shape == w.shape
    assert metrics.lte_estimate >= 0.0
    assert metrics.global_error_bound >= 0.0
    assert metrics.recommended_dt_ms > 0.0


def test_adex_step_uses_previous_voltage_for_adaptation() -> None:
    p = AdExParams()
    V = np.array([p.EL_mV + 5.0], dtype=float)
    w = np.array([1.0], dtype=float)
    state = AdExState(V_mV=V, w_pA=w, spiked=np.zeros_like(V, dtype=bool))
    out = adex_step(state, p, dt_ms=0.1, I_syn_pA=np.zeros(1), I_ext_pA=np.zeros(1))
    expected_dw = (p.a_nS * (V[0] - p.EL_mV) - w[0]) / p.tauw_ms
    expected_w = w[0] + 0.1 * expected_dw
    assert np.isclose(out.w_pA[0], expected_w)


def test_adex_step_rejects_invalid_inputs() -> None:
    p = AdExParams()
    V = np.array([p.EL_mV], dtype=float)
    w = np.array([0.0], dtype=float)
    state = AdExState(V_mV=V, w_pA=w, spiked=np.zeros_like(V, dtype=bool))
    with pytest.raises(ValueError, match="dt_ms must be positive"):
        adex_step(state, p, dt_ms=0.0, I_syn_pA=np.zeros(1), I_ext_pA=np.zeros(1))
    with pytest.raises(ValueError, match="dt_ms out of bounds"):
        adex_step(state, p, dt_ms=1.5, I_syn_pA=np.zeros(1), I_ext_pA=np.zeros(1))
    with pytest.raises(ValueError, match="I_syn_pA contains non-finite values"):
        adex_step(state, p, dt_ms=0.1, I_syn_pA=np.array([np.nan]), I_ext_pA=np.zeros(1))
    with pytest.raises(ValueError, match="I_ext_pA contains non-finite values"):
        adex_step(state, p, dt_ms=0.1, I_syn_pA=np.zeros(1), I_ext_pA=np.array([np.inf]))


def test_adex_step_adaptive_spike_reset_and_validation() -> None:
    p = AdExParams()
    state = AdExState(
        V_mV=np.array([p.Vpeak_mV + 20.0], dtype=float),
        w_pA=np.array([0.0], dtype=float),
        spiked=np.array([False]),
    )
    out = adex_step_adaptive(state, p, dt_ms=0.1, I_syn_pA=np.zeros(1), I_ext_pA=np.zeros(1))
    assert out.spiked[0]
    assert out.V_mV[0] == p.Vreset_mV
    assert out.w_pA[0] > 0.0
    with pytest.raises(ValueError, match="dt_ms must be positive"):
        adex_step_adaptive(state, p, dt_ms=0.0, I_syn_pA=np.zeros(1), I_ext_pA=np.zeros(1))
