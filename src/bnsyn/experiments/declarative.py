"""Declarative experiment execution from YAML configurations.

Provides YAML-driven experiment runner with schema validation.

References
----------
docs/LEGENDARY_QUICKSTART.md
schemas/experiment.schema.json
"""

from __future__ import annotations

import json
import hashlib
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import yaml  # type: ignore[import-untyped]

from bnsyn.experiments.emergence import run_emergence_to_disk
from bnsyn.numerics import compute_steps_exact
from bnsyn.schemas.experiment import BNSynExperimentConfig
from bnsyn.sim.network import run_simulation


def load_config(config_path: str | Path) -> BNSynExperimentConfig:
    """Load and validate experiment configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {config_path}: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML object, got {type(data).__name__}")

    try:
        return BNSynExperimentConfig(**data)
    except Exception as e:
        msg = f"❌ Config validation failed: {config_path}\n\nError: {e}"
        raise ValueError(msg) from e


def run_experiment(config: BNSynExperimentConfig) -> dict[str, Any]:
    """Run experiment from validated configuration."""
    results: dict[str, Any] = {
        "config": {
            "name": config.experiment.name,
            "version": config.experiment.version,
            "network_size": config.network.size,
            "duration_ms": config.simulation.duration_ms,
            "dt_ms": config.simulation.dt_ms,
            "external_current_pA": config.simulation.external_current_pA,
        },
        "runs": [],
    }

    steps = compute_steps_exact(config.simulation.duration_ms, config.simulation.dt_ms)

    for seed in config.experiment.seeds:
        if config.simulation.artifact_dir is None:
            metrics = run_simulation(
                steps=steps,
                dt_ms=config.simulation.dt_ms,
                seed=seed,
                N=config.network.size,
                external_current_pA=config.simulation.external_current_pA,
            )
            results["runs"].append({"seed": seed, "metrics": metrics})
        else:
            metrics, artifact_npz = run_emergence_to_disk(
                N=config.network.size,
                dt_ms=config.simulation.dt_ms,
                duration_ms=config.simulation.duration_ms,
                seed=seed,
                external_current_pA=config.simulation.external_current_pA,
                output_dir=config.simulation.artifact_dir,
            )
            results["runs"].append({"seed": seed, "metrics": metrics, "artifact_npz": artifact_npz})

    return results


def run_from_yaml(config_path: str | Path, output_path: str | Path | None = None) -> None:
    """Load config from YAML, run experiment, and save results."""
    config = load_config(config_path)
    print(f"✓ Config validated: {config.experiment.name} {config.experiment.version}")
    print(
        f"  Network: N={config.network.size}, "
        f"Duration: {config.simulation.duration_ms}ms, "
        f"dt: {config.simulation.dt_ms}ms"
    )
    print(f"  external_current_pA: {config.simulation.external_current_pA}")
    print(f"  Seeds: {len(config.experiment.seeds)} runs")

    results = run_experiment(config)

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(results, f, indent=2, sort_keys=True)
        print(f"✓ Results saved to {output_path}")
    else:
        print(json.dumps(results, indent=2, sort_keys=True))


def _write_grayscale_png(image: np.ndarray, output_path: Path) -> None:
    """Write a uint8 grayscale image to PNG without external plotting deps."""
    if image.dtype != np.uint8:
        raise ValueError("image must be uint8")
    if image.ndim != 2:
        raise ValueError("image must be 2-D")

    height, width = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    compressed = zlib.compress(raw, level=9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")
    output_path.write_bytes(png)


def _build_raster_image(spike_steps: np.ndarray, spike_neurons: np.ndarray, steps: int, n_neurons: int) -> np.ndarray:
    """Build monochrome raster image (white background, black spikes)."""
    width = max(steps, 2)
    height = max(n_neurons, 2)
    image = np.full((height, width), 255, dtype=np.uint8)
    for step, neuron in zip(spike_steps.tolist(), spike_neurons.tolist()):
        if 0 <= step < width and 0 <= neuron < height:
            image[height - 1 - neuron, step] = 0
    return image


def _build_rate_image(rate_trace_hz: np.ndarray, width: int = 1000, height: int = 300) -> np.ndarray:
    """Build monochrome line-plot style image for population-rate trace."""
    image = np.full((height, width), 255, dtype=np.uint8)
    if rate_trace_hz.size == 0:
        return image

    max_rate = float(np.max(rate_trace_hz))
    if max_rate <= 0:
        max_rate = 1.0

    sample_x = np.linspace(0, rate_trace_hz.size - 1, width)
    sampled = np.interp(sample_x, np.arange(rate_trace_hz.size), rate_trace_hz)

    for x, value in enumerate(sampled):
        y = int(round((1.0 - min(1.0, max(0.0, float(value) / max_rate))) * (height - 1)))
        image[y, x] = 0
        if y + 1 < height:
            image[y + 1, x] = 60
        if y > 0:
            image[y - 1, x] = 60

    image[height - 1, :] = 0
    return image


def _build_emergence_image(raster_image: np.ndarray, rate_image: np.ndarray) -> np.ndarray:
    """Build canonical emergence image as a composite of raster + rate traces."""
    if raster_image.ndim != 2 or rate_image.ndim != 2:
        raise ValueError("raster_image and rate_image must be 2D arrays")

    width = max(raster_image.shape[1], rate_image.shape[1])

    def _pad_to_width(image: np.ndarray, target_width: int) -> np.ndarray:
        if image.shape[1] == target_width:
            return image
        pad = np.full((image.shape[0], target_width - image.shape[1]), 255, dtype=np.uint8)
        return np.hstack((image, pad))

    raster = _pad_to_width(raster_image, width)
    rate = _pad_to_width(rate_image, width)
    separator = np.full((4, width), 180, dtype=np.uint8)
    return np.vstack((raster, separator, rate))


def _segment_avalanches(spike_counts: np.ndarray) -> tuple[list[int], list[int]]:
    """Segment contiguous nonzero bins into avalanche sizes and durations."""
    sizes: list[int] = []
    durations: list[int] = []
    current_size = 0
    current_duration = 0

    for count in spike_counts.tolist():
        count_i = int(count)
        if count_i > 0:
            current_size += count_i
            current_duration += 1
            continue
        if current_duration > 0:
            sizes.append(current_size)
            durations.append(current_duration)
            current_size = 0
            current_duration = 0

    if current_duration > 0:
        sizes.append(current_size)
        durations.append(current_duration)

    return sizes, durations


def _build_avalanche_report(
    *,
    seed: int,
    n_neurons: int,
    dt_ms: float,
    duration_ms: float,
    steps: int,
    spike_steps_per_step: np.ndarray,
    bin_width_steps: int = 1,
) -> dict[str, float | int | str | list[int]]:
    """Build deterministic avalanche analysis metrics from per-step spike counts."""
    if bin_width_steps <= 0:
        raise ValueError("bin_width_steps must be >= 1")
    if bin_width_steps != 1:
        usable = (spike_steps_per_step.size // bin_width_steps) * bin_width_steps
        rebinned = spike_steps_per_step[:usable].reshape(-1, bin_width_steps).sum(axis=1)
    else:
        rebinned = spike_steps_per_step

    sizes, durations = _segment_avalanches(np.asarray(rebinned, dtype=np.int64))
    nonempty_bins = int(np.count_nonzero(rebinned > 0))
    total_spikes = int(np.sum(rebinned))
    largest_size = max(sizes) if sizes else 0

    return {
        "schema_version": "1.0.0",
        "seed": seed,
        "N": int(n_neurons),
        "dt_ms": float(dt_ms),
        "duration_ms": float(duration_ms),
        "steps": int(steps),
        "bin_width_steps": int(bin_width_steps),
        "avalanche_count": int(len(sizes)),
        "active_bin_fraction": float(nonempty_bins / rebinned.size) if rebinned.size > 0 else 0.0,
        "size_mean": float(np.mean(sizes)) if sizes else 0.0,
        "size_max": int(largest_size),
        "duration_mean": float(np.mean(durations)) if durations else 0.0,
        "duration_max": int(max(durations)) if durations else 0,
        "sizes": sizes,
        "durations": durations,
        "nonempty_bins": nonempty_bins,
        "largest_avalanche_fraction": float(largest_size / total_spikes) if total_spikes > 0 else 0.0,
        "size_variance": float(np.var(sizes)) if sizes else 0.0,
        "duration_variance": float(np.var(durations)) if durations else 0.0,
    }


def _build_phase_space_report(
    *,
    seed: int,
    n_neurons: int,
    dt_ms: float,
    duration_ms: float,
    steps: int,
    rate_trace_hz: np.ndarray,
    sigma_trace: np.ndarray,
) -> dict[str, float | int | str | list[str] | dict[str, float]]:
    """Build deterministic state-space trajectory report from canonical traces."""
    rates = np.asarray(rate_trace_hz, dtype=np.float64)
    sigmas = np.asarray(sigma_trace, dtype=np.float64)

    if rates.size != sigmas.size:
        raise ValueError("rate_trace_hz and sigma_trace must have equal length")
    if rates.size != int(steps):
        raise ValueError("rate_trace_hz and sigma_trace length must equal steps")

    if rates.size == 0:
        rate_sigma_correlation = 0.0
        trajectory_length_l2 = 0.0
        rate_min = 0.0
        rate_max = 0.0
        sigma_min = 0.0
        sigma_max = 0.0
        centroid_rate = 0.0
        centroid_sigma = 0.0
        occupied_cell_fraction = 0.0
    else:
        rate_std = float(np.std(rates))
        sigma_std = float(np.std(sigmas))
        if rates.size < 2 or rate_std == 0.0 or sigma_std == 0.0:
            rate_sigma_correlation = 0.0
        else:
            rate_sigma_correlation = float(np.corrcoef(rates, sigmas)[0, 1])

        deltas_rate = np.diff(rates)
        deltas_sigma = np.diff(sigmas)
        trajectory_length_l2 = float(np.sum(np.sqrt(np.square(deltas_rate) + np.square(deltas_sigma))))

        rate_min = float(np.min(rates))
        rate_max = float(np.max(rates))
        sigma_min = float(np.min(sigmas))
        sigma_max = float(np.max(sigmas))

        centroid_rate = float(np.mean(rates))
        centroid_sigma = float(np.mean(sigmas))

        grid_size = 32
        visited = np.zeros((grid_size, grid_size), dtype=np.bool_)

        def _axis_idx(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
            if vmax <= vmin:
                return np.zeros(values.size, dtype=np.int64)
            scaled = (values - vmin) / (vmax - vmin)
            return np.clip((scaled * (grid_size - 1)).astype(np.int64), 0, grid_size - 1)

        x_idx = _axis_idx(rates, rate_min, rate_max)
        y_idx = _axis_idx(sigmas, sigma_min, sigma_max)
        visited[y_idx, x_idx] = True
        occupied_cell_fraction = float(np.count_nonzero(visited) / (grid_size * grid_size))

    return {
        "schema_version": "1.0.0",
        "seed": seed,
        "N": int(n_neurons),
        "dt_ms": float(dt_ms),
        "duration_ms": float(duration_ms),
        "steps": int(steps),
        "state_axes": ["population_rate_hz", "sigma"],
        "point_count": int(rates.size),
        "rate_mean_hz": float(np.mean(rates)) if rates.size else 0.0,
        "sigma_mean": float(np.mean(sigmas)) if sigmas.size else 0.0,
        "rate_sigma_correlation": rate_sigma_correlation,
        "trajectory_length_l2": trajectory_length_l2,
        "bounding_box": {
            "rate_min": rate_min,
            "rate_max": rate_max,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
        },
        "centroid": {
            "rate": centroid_rate,
            "sigma": centroid_sigma,
        },
        "occupied_cell_fraction": occupied_cell_fraction,
    }


def run_canonical_live_bundle(
    config_path: str | Path,
    artifact_dir: str | Path = "artifacts/canonical_run",
) -> dict[str, Any]:
    """Execute canonical profile and write deterministic live-run artifacts."""
    config = load_config(config_path)
    seed = int(config.experiment.seeds[0])

    metrics, artifact_npz = run_emergence_to_disk(
        N=config.network.size,
        dt_ms=config.simulation.dt_ms,
        duration_ms=config.simulation.duration_ms,
        seed=seed,
        external_current_pA=config.simulation.external_current_pA,
        output_dir=artifact_dir,
    )

    with np.load(artifact_npz) as data:
        spike_steps = np.asarray(data["spike_steps"], dtype=np.int64)
        sigma_trace = np.asarray(data["sigma_trace"], dtype=np.float64)
        rate_trace_hz = np.asarray(data["rate_trace_hz"], dtype=np.float64)
        spike_neurons = np.asarray(data["spike_neurons"], dtype=np.int64)
        steps = int(np.asarray(data["steps"]).item())
        n_neurons = int(np.asarray(data["N"]).item())

    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_metrics: dict[str, float | int] = {
        "spike_events": int(spike_steps.size),
        "rate_mean_hz": float(np.mean(rate_trace_hz)),
        "rate_peak_hz": float(np.max(rate_trace_hz)),
        "rate_variance": float(np.var(rate_trace_hz)),
        "sigma_mean": float(np.mean(sigma_trace)),
        "sigma_final": float(sigma_trace[-1]) if sigma_trace.size else 0.0,
        "sigma_variance": float(np.var(sigma_trace)),
        "seed": seed,
        "N": int(config.network.size),
        "steps": steps,
        "duration_ms": float(config.simulation.duration_ms),
        "dt_ms": float(config.simulation.dt_ms),
        "external_current_pA": float(config.simulation.external_current_pA),
    }
    summary_path = out_dir / "summary_metrics.json"
    summary_path.write_text(json.dumps(summary_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    spike_steps_per_step = np.bincount(spike_steps, minlength=steps) if steps > 0 else np.zeros(0, dtype=np.int64)
    active_steps = int(np.count_nonzero(spike_steps_per_step)) if steps > 0 else 0
    nonzero_rate_steps = int(np.count_nonzero(rate_trace_hz > 0.0))
    sigma_band = np.abs(sigma_trace - 1.0) <= 0.2
    sigma_distance = np.abs(sigma_trace - 1.0)

    criticality_report: dict[str, float | int | str] = {
        "schema_version": "1.0.0",
        "seed": seed,
        "N": int(config.network.size),
        "dt_ms": float(config.simulation.dt_ms),
        "duration_ms": float(config.simulation.duration_ms),
        "steps": steps,
        "sigma_mean": float(np.mean(sigma_trace)),
        "sigma_final": float(sigma_trace[-1]) if sigma_trace.size else 0.0,
        "sigma_variance": float(np.var(sigma_trace)),
        "rate_mean_hz": float(np.mean(rate_trace_hz)),
        "rate_peak_hz": float(np.max(rate_trace_hz)),
        "spike_events": int(spike_steps.size),
        "sigma_distance_from_1": float(np.mean(sigma_distance)) if sigma_distance.size else 0.0,
        "sigma_within_band_fraction": float(np.mean(sigma_band)) if sigma_band.size else 0.0,
        "active_steps_fraction": float(active_steps / steps) if steps > 0 else 0.0,
        "nonzero_rate_steps_fraction": float(nonzero_rate_steps / steps) if steps > 0 else 0.0,
        "rate_cv": float(np.std(rate_trace_hz) / np.mean(rate_trace_hz)) if np.mean(rate_trace_hz) > 0 else 0.0,
        "burstiness_proxy": float(np.var(spike_steps_per_step) / np.mean(spike_steps_per_step)) if np.mean(spike_steps_per_step) > 0 else 0.0,
    }
    criticality_report_path = out_dir / "criticality_report.json"
    criticality_report_path.write_text(
        json.dumps(criticality_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    avalanche_report = _build_avalanche_report(
        seed=seed,
        n_neurons=int(config.network.size),
        dt_ms=float(config.simulation.dt_ms),
        duration_ms=float(config.simulation.duration_ms),
        steps=steps,
        spike_steps_per_step=spike_steps_per_step,
        bin_width_steps=1,
    )
    avalanche_report_path = out_dir / "avalanche_report.json"
    avalanche_report_path.write_text(
        json.dumps(avalanche_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    phase_space_report = _build_phase_space_report(
        seed=seed,
        n_neurons=int(config.network.size),
        dt_ms=float(config.simulation.dt_ms),
        duration_ms=float(config.simulation.duration_ms),
        steps=steps,
        rate_trace_hz=rate_trace_hz,
        sigma_trace=sigma_trace,
    )
    phase_space_report_path = out_dir / "phase_space_report.json"
    phase_space_report_path.write_text(
        json.dumps(phase_space_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    raster_path = out_dir / "raster_plot.png"
    raster_image = _build_raster_image(spike_steps, spike_neurons, steps, n_neurons)
    _write_grayscale_png(raster_image, raster_path)

    rate_plot_path = out_dir / "population_rate_plot.png"
    rate_image = _build_rate_image(rate_trace_hz)
    _write_grayscale_png(rate_image, rate_plot_path)

    emergence_plot_path = out_dir / "emergence_plot.png"
    emergence_image = _build_emergence_image(raster_image, rate_image)
    _write_grayscale_png(emergence_image, emergence_plot_path)

    manifest = {
        "schema_version": "1.0.0",
        "cmd": "bnsyn run --profile canonical --plot --export-proof",
        "seed": seed,
        "steps": steps,
        "N": int(config.network.size),
        "dt_ms": float(config.simulation.dt_ms),
        "duration_ms": float(config.simulation.duration_ms),
        "artifacts": {
            "emergence_plot.png": hashlib.sha256(emergence_plot_path.read_bytes()).hexdigest(),
            "summary_metrics.json": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            "criticality_report.json": hashlib.sha256(criticality_report_path.read_bytes()).hexdigest(),
            "avalanche_report.json": hashlib.sha256(avalanche_report_path.read_bytes()).hexdigest(),
            "phase_space_report.json": hashlib.sha256(phase_space_report_path.read_bytes()).hexdigest(),
            "run_manifest.json": "self-unhashed",
            "raster_plot.png": hashlib.sha256(raster_path.read_bytes()).hexdigest(),
            "population_rate_plot.png": hashlib.sha256(rate_plot_path.read_bytes()).hexdigest(),
        },
    }
    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "artifact_dir": out_dir.as_posix(),
        "artifact_npz": artifact_npz,
        "run_manifest_path": manifest_path.as_posix(),
        "summary_metrics": summary_metrics,
        "summary_metrics_path": summary_path.as_posix(),
        "criticality_report": criticality_report,
        "criticality_report_path": criticality_report_path.as_posix(),
        "avalanche_report": avalanche_report,
        "avalanche_report_path": avalanche_report_path.as_posix(),
        "phase_space_report": phase_space_report,
        "phase_space_report_path": phase_space_report_path.as_posix(),
        "emergence_plot_path": emergence_plot_path.as_posix(),
        "raster_plot_path": raster_path.as_posix(),
        "population_rate_plot_path": rate_plot_path.as_posix(),
        "emergence_metrics": metrics,
    }
