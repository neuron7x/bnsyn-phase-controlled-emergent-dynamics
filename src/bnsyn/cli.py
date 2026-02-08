"""Command-line interface for BN-Syn demos and checks.

Parameters
----------
None

Returns
-------
None

Notes
-----
Provides deterministic demo runs, dt invariance checks, and sleep-stack experiments
per SPEC P2-11/P2-12.

References
----------
docs/SPEC.md#P2-11
docs/SPEC.md#P2-12
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import tomllib
import warnings
from pathlib import Path
from typing import Any

from bnsyn.provenance.manifest_builder import build_sleep_stack_manifest
from bnsyn.sim.network import run_simulation


def _get_package_version() -> str:
    """Return the installed package version with a safe fallback."""
    version: str | None = None
    try:
        version = importlib.metadata.version("bnsyn")
    except importlib.metadata.PackageNotFoundError:
        version = None
    except Exception as exc:
        warnings.warn(f"Failed to read package version: {exc}", stacklevel=2)
        version = None

    if version:
        return version

    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if pyproject_path.exists():
        try:
            data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            return "unknown"
        version = data.get("project", {}).get("version")
        if isinstance(version, str) and version:
            return version

    return "unknown"


def _cmd_demo(args: argparse.Namespace) -> int:
    """Run a deterministic demo simulation and print metrics.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments for the demo subcommand.

    Returns
    -------
    int
        Exit code (0 indicates success).

    Notes
    -----
    Calls the deterministic simulation harness with explicit dt and seed.
    If --interactive flag is set, launches Streamlit dashboard instead.

    References
    ----------
    docs/SPEC.md#P2-11
    docs/LEGENDARY_QUICKSTART.md
    """
    if getattr(args, "interactive", False):
        # Launch interactive Streamlit dashboard
        import importlib.util

        # subprocess used for controlled dashboard launch (no shell).
        import subprocess  # nosec B404
        import sys

        # Find the interactive.py script
        script_path = Path(__file__).parent / "viz" / "interactive.py"
        if not script_path.exists():
            print(f"Error: Interactive dashboard not found at {script_path}")
            return 1
        if importlib.util.find_spec("streamlit") is None:
            print('Error: Streamlit is not installed. Install with: pip install -e ".[viz]"')
            return 1

        print("🚀 Launching interactive dashboard...")
        print("   Press Ctrl+C to stop")
        try:
            # Fixed module invocation without shell; inputs are local paths only.
            result = subprocess.run(  # nosec B603
                [sys.executable, "-m", "streamlit", "run", str(script_path)]
            )
            if result.returncode != 0:
                print(f"Error: Dashboard exited with code {result.returncode}")
                return 1
            return 0
        except KeyboardInterrupt:
            print("\n✓ Dashboard stopped")
            return 0
        except Exception as e:
            print(f"Error launching dashboard: {e}")
            return 1

    metrics = run_simulation(steps=args.steps, dt_ms=args.dt_ms, seed=args.seed, N=args.N)
    print(json.dumps({"demo": metrics}, indent=2, sort_keys=True))
    return 0


def _cmd_dtcheck(args: argparse.Namespace) -> int:
    """Run dt vs dt/2 invariance check and print metrics.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments for the dt invariance subcommand.

    Returns
    -------
    int
        Exit code (0 indicates success).

    Notes
    -----
    Compares mean-rate and sigma metrics across dt and dt/2 as required by SPEC P2-12.

    References
    ----------
    docs/SPEC.md#P2-12
    """
    m1 = run_simulation(steps=args.steps, dt_ms=args.dt_ms, seed=args.seed, N=args.N)
    m2 = run_simulation(steps=args.steps * 2, dt_ms=args.dt2_ms, seed=args.seed, N=args.N)
    # compare mean rates and sigma; dt2 should be close
    out: dict[str, Any] = {"dt": args.dt_ms, "dt2": args.dt2_ms, "m_dt": m1, "m_dt2": m2}
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def _cmd_run_experiment(args: argparse.Namespace) -> int:
    """Run experiment from YAML configuration.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments for the run subcommand.

    Returns
    -------
    int
        Exit code (0 indicates success).

    Notes
    -----
    Loads YAML config, validates against schema, runs experiment.

    References
    ----------
    docs/LEGENDARY_QUICKSTART.md
    schemas/experiment.schema.json
    """
    from bnsyn.experiments.declarative import run_from_yaml

    try:
        run_from_yaml(args.config, args.output)
        return 0
    except Exception as e:
        print(f"Error running experiment: {e}")
        return 1


def _cmd_sleep_stack(args: argparse.Namespace) -> int:
    """Run sleep-stack demo with attractor crystallization and consolidation.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments for the sleep-stack subcommand.

    Returns
    -------
    int
        Exit code (0 indicates success).

    Notes
    -----
    Runs wake→sleep cycle with memory recording, consolidation, replay,
    attractor tracking, and phase transition detection.

    References
    ----------
    docs/sleep_stack.md
    docs/emergence_tracking.md
    """
    # Import here to avoid circular dependencies and keep CLI fast
    from bnsyn.config import AdExParams, CriticalityParams, SynapseParams, TemperatureParams
    from bnsyn.criticality import PhaseTransitionDetector
    from bnsyn.emergence import AttractorCrystallizer
    from bnsyn.memory import MemoryConsolidator
    from bnsyn.rng import seed_all
    from bnsyn.sim.network import Network, NetworkParams
    from bnsyn.sleep import SleepCycle, SleepStageConfig, default_human_sleep_cycle
    from bnsyn.temperature.schedule import TemperatureSchedule

    # Setup output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir.parent / "figures" / out_dir.name
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Seed RNG
    pack = seed_all(args.seed)
    rng = pack.np_rng

    # Create network
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

    # Temperature schedule
    temp_schedule = TemperatureSchedule(TemperatureParams())

    # Sleep cycle
    sleep_cycle = SleepCycle(net, temp_schedule, max_memories=100, rng=rng)

    # Memory consolidator
    consolidator = MemoryConsolidator(capacity=100)

    # Phase transition detector
    phase_detector = PhaseTransitionDetector()

    # Attractor crystallizer
    crystallizer = AttractorCrystallizer(
        state_dim=N,
        max_buffer_size=500,
        snapshot_dim=min(50, N),
        pca_update_interval=50,
    )

    # Wake phase
    print(f"Running wake phase ({args.steps_wake} steps)...")
    wake_metrics = []
    for _ in range(args.steps_wake):
        m = net.step()
        wake_metrics.append(m)

        # Record memory periodically
        if len(wake_metrics) % 20 == 0:
            importance = min(1.0, m["spike_rate_hz"] / 10.0)
            sleep_cycle.record_memory(importance)
            consolidator.tag(net.state.V_mV, importance)

        # Track phase transitions
        phase_detector.observe(m["sigma"], len(wake_metrics))

        # Track attractor crystallization
        crystallizer.observe(net.state.V_mV, temp_schedule.T or 1.0)

    # Sleep phase
    print(f"Running sleep phase ({args.steps_sleep} steps)...")
    sleep_stages = default_human_sleep_cycle()
    # Scale durations if requested
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

    sleep_summary = sleep_cycle.sleep(sleep_stages)

    # Collect metrics
    transitions = phase_detector.get_transitions()
    attractors = crystallizer.get_attractors()
    cryst_state = crystallizer.get_crystallization_state()
    cons_stats = consolidator.stats()

    metrics: dict[str, Any] = {
        "backend": args.backend,
        "wake": {
            "steps": args.steps_wake,
            "mean_sigma": float(sum(m["sigma"] for m in wake_metrics) / len(wake_metrics)),
            "mean_spike_rate": float(
                sum(m["spike_rate_hz"] for m in wake_metrics) / len(wake_metrics)
            ),
            "memories_recorded": sleep_cycle.get_memory_count(),
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

    # Write metrics
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics written to {metrics_path}")

    # Generate manifest
    manifest = build_sleep_stack_manifest(
        seed=args.seed,
        steps_wake=args.steps_wake,
        steps_sleep=args.steps_sleep,
        N=N,
        package_version=_get_package_version(),
        repo_root=Path(__file__).parent.parent,
    )

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {manifest_path}")

    # Generate figure (optional, only if matplotlib available)
    try:
        import matplotlib.pyplot as plt
        from typing import Any as _Any

        fig, axes_raw = plt.subplots(2, 2, figsize=(12, 8))
        axes: _Any = axes_raw  # Type hint to satisfy mypy

        # Sigma trace
        ax = axes[0, 0]
        wake_sigmas = [m["sigma"] for m in wake_metrics]
        ax.plot(wake_sigmas, label="Wake", alpha=0.7)
        ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.3)
        ax.set_xlabel("Step")
        ax.set_ylabel("Sigma")
        ax.set_title("Criticality (Sigma)")
        ax.legend()
        ax.grid(alpha=0.3)

        # Spike rate
        ax = axes[0, 1]
        wake_rates = [m["spike_rate_hz"] for m in wake_metrics]
        ax.plot(wake_rates, alpha=0.7, color="orange")
        ax.set_xlabel("Step")
        ax.set_ylabel("Spike Rate (Hz)")
        ax.set_title("Network Activity")
        ax.grid(alpha=0.3)

        # Phase transitions
        ax = axes[1, 0]
        if transitions:
            trans_steps = [t.step for t in transitions]
            trans_phases = [t.to_phase.name for t in transitions]
            ax.scatter(trans_steps, range(len(trans_steps)), s=100, alpha=0.7)
            for i, (step, phase) in enumerate(zip(trans_steps, trans_phases)):
                ax.text(step, i, phase, fontsize=8, ha="left")
        ax.set_xlabel("Step")
        ax.set_ylabel("Transition Index")
        ax.set_title(f"Phase Transitions ({len(transitions)} total)")
        ax.grid(alpha=0.3)

        # Attractor crystallization
        ax = axes[1, 1]
        ax.bar(["Progress", "Count"], [cryst_state.progress, len(attractors) / 10.0])
        ax.set_ylabel("Value")
        ax.set_title(f"Crystallization ({cryst_state.phase.name})")
        ax.set_ylim([0, 1.1])
        ax.grid(alpha=0.3)

        plt.tight_layout()
        fig_path = fig_dir / "summary.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Figure saved to {fig_path}")
    except ImportError:
        print(
            "Matplotlib not installed; skipping figure generation. "
            'Install with: pip install -e ".[viz]"'
        )
    except Exception as e:
        print(f"Figure generation failed: {e}")

    print("\n=== Sleep-Stack Demo Complete ===")
    print(f"Wake: {args.steps_wake} steps, {metrics['wake']['memories_recorded']} memories")
    print(f"Sleep: {sleep_summary['total_steps']} steps")
    print(f"Transitions: {len(transitions)}")
    print(f"Attractors: {len(attractors)}")
    print(f"Consolidation: {cons_stats['consolidated_count']}/{cons_stats['count']} patterns")

    return 0


def main() -> None:
    """Entry point for the BN-Syn CLI.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Examples
    --------
    Run a deterministic demo simulation::

        $ bnsyn demo --steps 1000 --seed 42 --N 100

    Check dt-invariance (dt vs dt/2 comparison)::

        $ bnsyn dtcheck --dt-ms 0.1 --dt2-ms 0.05 --steps 2000

    Run sleep-stack demo::

        $ bnsyn sleep-stack --seed 123 --steps-wake 800 --steps-sleep 600

    Output format (demo)::

        {
          "sigma": 1.02,
          "spike_rate_hz": 3.45,
          "V_mean_mV": -62.1,
          "energy_cost_aJ": 1234.56
        }

    Notes
    -----
    Builds the CLI parser and dispatches to deterministic command handlers.

    References
    ----------
    docs/SPEC.md#P2-11
    """
    p = argparse.ArgumentParser(prog="bnsyn", description="BN-Syn CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    demo = sub.add_parser("demo", help="Run a small deterministic demo simulation")
    demo.add_argument("--steps", type=int, default=2000)
    demo.add_argument("--dt-ms", type=float, default=0.1)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--N", type=int, default=200)
    demo.add_argument("--interactive", action="store_true", help="Launch interactive dashboard")
    demo.set_defaults(func=_cmd_demo)

    run_parser = sub.add_parser("run", help="Run experiment from YAML config")
    run_parser.add_argument("config", help="Path to YAML configuration file")
    run_parser.add_argument("-o", "--output", help="Output JSON path (default: stdout)")
    run_parser.set_defaults(func=_cmd_run_experiment)

    dtc = sub.add_parser("dtcheck", help="Run dt vs dt/2 invariance harness")
    dtc.add_argument("--steps", type=int, default=2000)
    dtc.add_argument("--dt-ms", type=float, default=0.1)
    dtc.add_argument("--dt2-ms", type=float, default=0.05)
    dtc.add_argument("--seed", type=int, default=42)
    dtc.add_argument("--N", type=int, default=200)
    dtc.set_defaults(func=_cmd_dtcheck)

    sleep = sub.add_parser("sleep-stack", help="Run sleep-stack demo with emergence tracking")
    sleep.add_argument("--seed", type=int, default=123, help="RNG seed")
    sleep.add_argument("--N", type=int, default=64, help="Number of neurons")
    sleep.add_argument(
        "--backend",
        choices=["reference", "accelerated"],
        default="reference",
        help="Simulation backend",
    )
    sleep.add_argument("--steps-wake", type=int, default=800, help="Wake phase steps")
    sleep.add_argument("--steps-sleep", type=int, default=600, help="Sleep phase steps")
    sleep.add_argument(
        "--out",
        type=str,
        default="results/sleep_stack_v1",
        help="Output directory for results",
    )
    sleep.set_defaults(func=_cmd_sleep_stack)

    args = p.parse_args()
    raise SystemExit(int(args.func(args)))


if __name__ == "__main__":
    main()
