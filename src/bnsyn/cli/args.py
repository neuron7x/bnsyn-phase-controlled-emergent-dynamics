"""Argument parser construction for the BN-Syn CLI."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level CLI parser."""
    parser = argparse.ArgumentParser(prog="bnsyn", description="BN-Syn CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    demo = sub.add_parser("demo", help="Run a small deterministic demo simulation")
    demo.add_argument("--steps", type=int, default=2000)
    demo.add_argument("--dt-ms", type=float, default=0.1)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--N", type=int, default=200)
    demo.add_argument("--interactive", action="store_true", help="Launch interactive dashboard")

    run_parser = sub.add_parser("run", help="Run experiment from YAML config")
    run_parser.add_argument("config", help="Path to YAML configuration file")
    run_parser.add_argument("-o", "--output", help="Output JSON path (default: stdout)")

    dtc = sub.add_parser("dtcheck", help="Run dt vs dt/2 invariance harness")
    dtc.add_argument("--steps", type=int, default=2000)
    dtc.add_argument("--dt-ms", type=float, default=0.1)
    dtc.add_argument("--dt2-ms", type=float, default=0.05)
    dtc.add_argument("--seed", type=int, default=42)
    dtc.add_argument("--N", type=int, default=200)

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
    return parser
