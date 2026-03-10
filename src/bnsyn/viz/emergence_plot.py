"""Plot emergence artifacts from ``run_<seed>.npz`` outputs."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np


_PLOT_REQUIRED_FIELDS = {
    "format_version",
    "spike_steps",
    "spike_neurons",
    "sigma_trace",
    "rate_trace_hz",
    "dt_ms",
    "steps",
    "N",
    "seed",
    "external_current_pA",
}
EXPECTED_ARTIFACT_FORMAT_VERSION = "1.1.0"

_plt: Any | None = None


def _load_pyplot() -> Any:
    """Load matplotlib.pyplot lazily for optional visualization support."""
    global _plt
    if _plt is not None:
        return _plt
    try:
        _plt = importlib.import_module("matplotlib.pyplot")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            'Visualization requires matplotlib. Install with: pip install -e ".[viz]"'
        ) from exc
    return _plt


def _validate_npz_contract(data: np.lib.npyio.NpzFile) -> None:
    """Validate required emergence artifact fields and core shape invariants."""
    missing = _PLOT_REQUIRED_FIELDS - set(data.files)
    if missing:
        raise ValueError(f"NPZ artifact missing required fields: {sorted(missing)}")

    format_version = str(data["format_version"])
    if format_version != EXPECTED_ARTIFACT_FORMAT_VERSION:
        raise RuntimeError(
            "Artifact format mismatch: "
            f"expected {EXPECTED_ARTIFACT_FORMAT_VERSION}, got {format_version}"
        )

    spike_steps = data["spike_steps"]
    spike_neurons = data["spike_neurons"]
    rate_trace = data["rate_trace_hz"]
    sigma_trace = data["sigma_trace"]
    steps = int(data["steps"])

    if spike_steps.shape != spike_neurons.shape:
        raise ValueError("NPZ artifact invalid: spike_steps and spike_neurons shapes must match")
    if rate_trace.shape != sigma_trace.shape:
        raise ValueError("NPZ artifact invalid: rate_trace_hz and sigma_trace shapes must match")
    if rate_trace.ndim != 1:
        raise ValueError("NPZ artifact invalid: rate_trace_hz must be 1D")
    if sigma_trace.ndim != 1:
        raise ValueError("NPZ artifact invalid: sigma_trace must be 1D")
    if rate_trace.shape[0] != steps:
        raise ValueError("NPZ artifact invalid: trace lengths must match steps")


def plot_emergence_npz(npz_path: str | Path, output_path: str | Path) -> Path:
    """Render raster/population-rate/sigma plots from a saved emergence NPZ."""
    src = Path(npz_path)
    out = Path(output_path)

    with np.load(src) as data:
        _validate_npz_contract(data)
        spike_steps = data["spike_steps"]
        spike_neurons = data["spike_neurons"]
        rate_trace = data["rate_trace_hz"]
        sigma_trace = data["sigma_trace"]
        dt_ms = float(data["dt_ms"])
        n = int(data["N"])
        current = float(data["external_current_pA"])

    time_ms = np.arange(rate_trace.shape[0], dtype=np.float64) * dt_ms

    plt = _load_pyplot()
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    if spike_steps.size > 0:
        axes[0].scatter(spike_steps * dt_ms, spike_neurons, s=2, alpha=0.7)
    axes[0].set_ylabel("neuron")
    axes[0].set_title(f"Raster (N={n}, I_ext={current:.1f} pA)")

    axes[1].plot(time_ms, rate_trace)
    axes[1].set_ylabel("rate (Hz)")
    axes[1].set_title("Population rate")

    axes[2].plot(time_ms, sigma_trace)
    axes[2].set_ylabel("sigma")
    axes[2].set_xlabel("time (ms)")
    axes[2].set_title("Branching ratio")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
