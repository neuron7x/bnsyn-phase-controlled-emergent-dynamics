import numpy as np
from bnsyn.config import AdExParams
from bnsyn.neuron.adex import AdExState, adex_step


def test_adex_step_spikes_and_resets() -> None:
    p = AdExParams()
    V = np.array([p.Vpeak_mV + 1.0, p.EL_mV], dtype=float)
    w = np.zeros_like(V)
    s = AdExState(V_mV=V, w_pA=w, spiked=np.zeros_like(V, dtype=bool))
    out = adex_step(s, p, dt_ms=0.1, I_syn_pA=np.zeros(2), I_ext_pA=np.zeros(2))
    assert out.spiked[0]
    assert out.V_mV[0] == p.Vreset_mV
    assert out.w_pA[0] > 0
