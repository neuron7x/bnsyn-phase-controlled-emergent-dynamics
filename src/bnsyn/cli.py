"""Command-line interface for BN-Syn demos and checks.

Parameters
----------
None

Returns
-------
None

Determinism
-----------
Deterministic under fixed seeds and fixed parameters.

SPEC
----
SPEC.md §P2-11, §P2-12

Claims
------
CLM-0026
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from bnsyn.sim.network import run_simulation


def _cmd_demo(args: argparse.Namespace) -> int:
    """Run a deterministic demo simulation and print metrics.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Exit code.

    Determinism
    -----------
    Deterministic under fixed seed and fixed parameters.

    SPEC
    ----
    SPEC.md §P2-11, §P2-12

    Claims
    ------
    CLM-0026
    """
    metrics = run_simulation(steps=args.steps, dt_ms=args.dt_ms, seed=args.seed, N=args.N)
    print(json.dumps({"demo": metrics}, indent=2, sort_keys=True))
    return 0


def _cmd_dtcheck(args: argparse.Namespace) -> int:
    """Run dt vs dt/2 invariance check and print metrics.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Exit code.

    Determinism
    -----------
    Deterministic under fixed seed and fixed parameters.

    SPEC
    ----
    SPEC.md §P2-11, §P2-12

    Claims
    ------
    CLM-0026
    """
    m1 = run_simulation(steps=args.steps, dt_ms=args.dt_ms, seed=args.seed, N=args.N)
    m2 = run_simulation(steps=args.steps * 2, dt_ms=args.dt2_ms, seed=args.seed, N=args.N)
    # compare mean rates and sigma; dt2 should be close
    out: dict[str, Any] = {"dt": args.dt_ms, "dt2": args.dt2_ms, "m_dt": m1, "m_dt2": m2}
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def main() -> None:
    """Entry point for the BN-Syn CLI.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Determinism
    -----------
    Deterministic under fixed CLI inputs.

    SPEC
    ----
    SPEC.md §P2-12

    Claims
    ------
    CLM-0026
    """
    p = argparse.ArgumentParser(prog="bnsyn", description="BN-Syn CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    demo = sub.add_parser("demo", help="Run a small deterministic demo simulation")
    demo.add_argument("--steps", type=int, default=2000)
    demo.add_argument("--dt-ms", type=float, default=0.1)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--N", type=int, default=200)
    demo.set_defaults(func=_cmd_demo)

    dtc = sub.add_parser("dtcheck", help="Run dt vs dt/2 invariance harness")
    dtc.add_argument("--steps", type=int, default=2000)
    dtc.add_argument("--dt-ms", type=float, default=0.1)
    dtc.add_argument("--dt2-ms", type=float, default=0.05)
    dtc.add_argument("--seed", type=int, default=42)
    dtc.add_argument("--N", type=int, default=200)
    dtc.set_defaults(func=_cmd_dtcheck)

    args = p.parse_args()
    raise SystemExit(int(args.func(args)))


if __name__ == "__main__":
    main()
