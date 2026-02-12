"""CLI command handlers and sleep-stack execution helpers."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess  # nosec B404
import sys
from pathlib import Path
from typing import Any, cast

from bnsyn.provenance.manifest_builder import build_sleep_stack_manifest
from bnsyn.sim.network import run_simulation

from .version import get_package_version


def cmd_demo(args: argparse.Namespace) -> int:
    """Execute the deterministic demo command."""
    if getattr(args, "interactive", False):
        return _launch_interactive_dashboard()
    metrics = run_simulation(steps=args.steps, dt_ms=args.dt_ms, seed=args.seed, N=args.N)
    print(json.dumps({"demo": metrics}, indent=2, sort_keys=True))
    return 0


def _launch_interactive_dashboard() -> int:
    script_path = Path(__file__).resolve().parents[1] / "viz" / "interactive.py"
    if not script_path.exists():
        print(f"Error: Interactive dashboard not found at {script_path}")
        return 1
    if importlib.util.find_spec("streamlit") is None:
        print('Error: Streamlit is not installed. Install with: pip install -e ".[viz]"')
        return 1

    print("🚀 Launching interactive dashboard...")
    print("   Press Ctrl+C to stop")
    try:
        result = subprocess.run([sys.executable, "-m", "streamlit", "run", str(script_path)])  # nosec B603
        if result.returncode != 0:
            print(f"Error: Dashboard exited with code {result.returncode}")
            return 1
        return 0
    except KeyboardInterrupt:
        print("\n✓ Dashboard stopped")
        return 0
    except Exception as exc:
        print(f"Error launching dashboard: {exc}")
        return 1


def cmd_dtcheck(args: argparse.Namespace) -> int:
    """Execute dt invariance diagnostics."""
    m1 = run_simulation(steps=args.steps, dt_ms=args.dt_ms, seed=args.seed, N=args.N)
    m2 = run_simulation(steps=args.steps * 2, dt_ms=args.dt2_ms, seed=args.seed, N=args.N)
    out: dict[str, Any] = {"dt": args.dt_ms, "dt2": args.dt2_ms, "m_dt": m1, "m_dt2": m2}
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def cmd_run_experiment(args: argparse.Namespace) -> int:
    """Execute declarative experiment run from YAML config."""
    from bnsyn.experiments.declarative import run_from_yaml

    try:
        run_from_yaml(args.config, args.output)
        return 0
    except Exception as exc:
        print(f"Error running experiment: {exc}")
        return 1


def cmd_sleep_stack(args: argparse.Namespace) -> int:
    """Execute wake/sleep-stack simulation and artifact generation."""
    context = _build_sleep_stack_context(args)
    wake_metrics = _run_wake_phase(args, context)
    sleep_summary = _run_sleep_phase(args, context)
    metrics = _collect_sleep_stack_metrics(args, context, wake_metrics, sleep_summary)
    _write_sleep_stack_artifacts(args, context, metrics)
    _render_sleep_stack_figure(context, wake_metrics, metrics)
    _print_sleep_stack_summary(args, metrics)
    return 0


def _build_sleep_stack_context(args: argparse.Namespace) -> dict[str, Any]:
    from bnsyn.config import AdExParams, CriticalityParams, SynapseParams, TemperatureParams
    from bnsyn.criticality import PhaseTransitionDetector
    from bnsyn.emergence import AttractorCrystallizer
    from bnsyn.memory import MemoryConsolidator
    from bnsyn.rng import seed_all
    from bnsyn.sim.network import Network, NetworkParams
    from bnsyn.sleep import SleepCycle
    from bnsyn.temperature.schedule import TemperatureSchedule

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir.parent / "figures" / out_dir.name
    fig_dir.mkdir(parents=True, exist_ok=True)

    rng = seed_all(args.seed).np_rng
    N = int(args.N)
    nparams = NetworkParams(N=N)
    net = Network(
        nparams,
        AdExParams(),
        SynapseParams(),
        CriticalityParams(),
        dt_ms=0.5,
        rng=rng,
        backend=args.backend,
    )
    temp_schedule = TemperatureSchedule(TemperatureParams())
    return {
        "N": N,
        "out_dir": out_dir,
        "fig_dir": fig_dir,
        "net": net,
        "temp_schedule": temp_schedule,
        "sleep_cycle": SleepCycle(net, temp_schedule, max_memories=100, rng=rng),
        "consolidator": MemoryConsolidator(capacity=100),
        "phase_detector": PhaseTransitionDetector(),
        "crystallizer": AttractorCrystallizer(
            state_dim=N,
            max_buffer_size=500,
            snapshot_dim=min(50, N),
            pca_update_interval=50,
        ),
    }


def _run_wake_phase(args: argparse.Namespace, context: dict[str, Any]) -> list[dict[str, float]]:
    print(f"Running wake phase ({args.steps_wake} steps)...")
    wake_metrics: list[dict[str, float]] = []
    net = context["net"]
    temp_schedule = context["temp_schedule"]
    sleep_cycle = context["sleep_cycle"]
    consolidator = context["consolidator"]
    phase_detector = context["phase_detector"]
    crystallizer = context["crystallizer"]

    for _ in range(args.steps_wake):
        metrics = net.step()
        wake_metrics.append(metrics)
        if len(wake_metrics) % 20 == 0:
            importance = min(1.0, metrics["spike_rate_hz"] / 10.0)
            sleep_cycle.record_memory(importance)
            consolidator.tag(net.state.V_mV, importance)
        phase_detector.observe(metrics["sigma"], len(wake_metrics))
        crystallizer.observe(net.state.V_mV, temp_schedule.T or 1.0)
    return wake_metrics


def _run_sleep_phase(args: argparse.Namespace, context: dict[str, Any]) -> dict[str, Any]:
    from bnsyn.sleep import SleepStageConfig, default_human_sleep_cycle

    print(f"Running sleep phase ({args.steps_sleep} steps)...")
    sleep_stages = default_human_sleep_cycle()
    if args.steps_sleep != 600:
        scale = args.steps_sleep / 450
        sleep_stages = [
            SleepStageConfig(
                stage=stage.stage,
                duration_steps=int(stage.duration_steps * scale),
                temperature_range=stage.temperature_range,
                replay_active=stage.replay_active,
                replay_noise=stage.replay_noise,
            )
            for stage in sleep_stages
        ]
    return cast(dict[str, Any], context["sleep_cycle"].sleep(sleep_stages))


def _collect_sleep_stack_metrics(
    args: argparse.Namespace,
    context: dict[str, Any],
    wake_metrics: list[dict[str, float]],
    sleep_summary: dict[str, Any],
) -> dict[str, Any]:
    transitions = context["phase_detector"].get_transitions()
    attractors = context["crystallizer"].get_attractors()
    cryst_state = context["crystallizer"].get_crystallization_state()
    cons_stats = context["consolidator"].stats()
    return {
        "backend": args.backend,
        "wake": {
            "steps": args.steps_wake,
            "mean_sigma": float(sum(m["sigma"] for m in wake_metrics) / len(wake_metrics)),
            "mean_spike_rate": float(
                sum(m["spike_rate_hz"] for m in wake_metrics) / len(wake_metrics)
            ),
            "memories_recorded": context["sleep_cycle"].get_memory_count(),
        },
        "sleep": sleep_summary,
        "transitions": [
            {
                "step": t.step,
                "from": t.from_phase.name,
                "to": t.to_phase.name,
                "sigma_before": t.sigma_before,
                "sigma_after": t.sigma_after,
                "sharpness": t.sharpness,
            }
            for t in transitions
        ],
        "attractors": {
            "count": len(attractors),
            "crystallization_progress": cryst_state.progress,
            "phase": cryst_state.phase.name,
        },
        "consolidation": cons_stats,
    }


def _write_sleep_stack_artifacts(args: argparse.Namespace, context: dict[str, Any], metrics: dict[str, Any]) -> None:
    out_dir: Path = context["out_dir"]
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Metrics written to {metrics_path}")

    manifest = build_sleep_stack_manifest(
        seed=args.seed,
        steps_wake=args.steps_wake,
        steps_sleep=args.steps_sleep,
        N=context["N"],
        package_version=get_package_version(),
        repo_root=Path(__file__).resolve().parents[2],
    )
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Manifest written to {manifest_path}")




def _plot_sigma_panel(axes: Any, wake_metrics: list[dict[str, float]]) -> None:
    ax = axes[0, 0]
    wake_sigmas = [m["sigma"] for m in wake_metrics]
    ax.plot(wake_sigmas, label="Wake", alpha=0.7)
    ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Step")
    ax.set_ylabel("Sigma")
    ax.set_title("Criticality (Sigma)")
    ax.legend()
    ax.grid(alpha=0.3)


def _plot_rate_panel(axes: Any, wake_metrics: list[dict[str, float]]) -> None:
    ax = axes[0, 1]
    wake_rates = [m["spike_rate_hz"] for m in wake_metrics]
    ax.plot(wake_rates, alpha=0.7, color="orange")
    ax.set_xlabel("Step")
    ax.set_ylabel("Spike Rate (Hz)")
    ax.set_title("Network Activity")
    ax.grid(alpha=0.3)


def _plot_transitions_panel(axes: Any, transitions: list[dict[str, Any]]) -> None:
    ax = axes[1, 0]
    if transitions:
        trans_steps = [t["step"] for t in transitions]
        trans_phases = [t["to"] for t in transitions]
        ax.scatter(trans_steps, range(len(trans_steps)), s=100, alpha=0.7)
        for i, (step, phase) in enumerate(zip(trans_steps, trans_phases, strict=False)):
            ax.text(step, i, phase, fontsize=8, ha="left")
    ax.set_xlabel("Step")
    ax.set_ylabel("Transition Index")
    ax.set_title(f"Phase Transitions ({len(transitions)} total)")
    ax.grid(alpha=0.3)


def _plot_attractor_panel(axes: Any, metrics: dict[str, Any]) -> None:
    ax = axes[1, 1]
    ax.bar(
        ["Progress", "Count"],
        [metrics["attractors"]["crystallization_progress"], metrics["attractors"]["count"] / 10.0],
    )
    ax.set_ylabel("Value")
    ax.set_title(f"Crystallization ({metrics['attractors']['phase']})")
    ax.set_ylim([0, 1.1])
    ax.grid(alpha=0.3)

def _render_sleep_stack_figure(
    context: dict[str, Any], wake_metrics: list[dict[str, float]], metrics: dict[str, Any]
) -> None:
    try:
        import matplotlib.pyplot as plt
        from typing import Any as _Any

        fig, axes_raw = plt.subplots(2, 2, figsize=(12, 8))
        axes: _Any = axes_raw
        transitions = metrics["transitions"]
        _plot_sigma_panel(axes, wake_metrics)
        _plot_rate_panel(axes, wake_metrics)
        _plot_transitions_panel(axes, transitions)
        _plot_attractor_panel(axes, metrics)

        plt.tight_layout()
        fig_path = context["fig_dir"] / "summary.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Figure saved to {fig_path}")
    except ImportError:
        print(
            "Matplotlib not installed; skipping figure generation. "
            'Install with: pip install -e ".[viz]"'
        )
    except Exception as exc:
        print(f"Figure generation failed: {exc}")


def _print_sleep_stack_summary(args: argparse.Namespace, metrics: dict[str, Any]) -> None:
    print("\n=== Sleep-Stack Demo Complete ===")
    print(f"Wake: {args.steps_wake} steps, {metrics['wake']['memories_recorded']} memories")
    print(f"Sleep: {metrics['sleep']['total_steps']} steps")
    print(f"Transitions: {len(metrics['transitions'])}")
    print(f"Attractors: {metrics['attractors']['count']}")
    cons = metrics["consolidation"]
    print(f"Consolidation: {cons['consolidated_count']}/{cons['count']} patterns")
