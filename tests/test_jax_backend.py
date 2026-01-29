"""Tests for the optional JAX backend."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from types import ModuleType

import numpy as np
import pytest


def _load_module_with_jnp(fake_jnp: ModuleType) -> ModuleType:
    module_name = "bnsyn.production.jax_backend"
    sys.modules["jax"] = ModuleType("jax")
    sys.modules["jax.numpy"] = fake_jnp
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)


def test_jax_backend_import_error_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "bnsyn.production.jax_backend"

    original_import = importlib.import_module

    def fake_import(name: str, *args: object, **kwargs: object) -> ModuleType:
        if name == "jax.numpy":
            raise ImportError("no jax")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    sys.modules.pop(module_name, None)
    sys.modules.pop("jax.numpy", None)
    sys.modules.pop("jax", None)

    spec = importlib.util.find_spec(module_name)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(ImportError, match="JAX is not installed"):
        spec.loader.exec_module(module)


def test_adex_step_jax_with_numpy_shim(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_jnp = ModuleType("jax.numpy")
    fake_jnp.exp = np.exp
    fake_jnp.where = np.where

    module = _load_module_with_jnp(fake_jnp)

    V = np.array([-55.0, -40.0], dtype=float)
    w = np.array([0.0, 0.0], dtype=float)
    input_current = np.array([0.0, 500.0], dtype=float)

    V_new, w_new, spikes = module.adex_step_jax(
        V,
        w,
        input_current,
        C=200.0,
        gL=10.0,
        EL=-65.0,
        VT=-50.0,
        DeltaT=2.0,
        tau_w=100.0,
        a=2.0,
        b=50.0,
        V_reset=-65.0,
        V_spike=-40.0,
        dt=0.1,
    )

    assert V_new.shape == V.shape
    assert w_new.shape == w.shape
    assert spikes.shape == V.shape
    assert spikes.dtype == bool
    assert V_new[1] == -65.0
    assert w_new[1] > w[1]

    sys.modules.pop("bnsyn.production.jax_backend", None)
    sys.modules.pop("jax.numpy", None)
    sys.modules.pop("jax", None)
