"""Command-line interface for BN-Syn demos and checks."""

from __future__ import annotations

from .args import build_parser
from .commands import (
    cmd_demo as _cmd_demo,
    cmd_dtcheck as _cmd_dtcheck,
    cmd_run_experiment as _cmd_run_experiment,
    cmd_sleep_stack as _cmd_sleep_stack,
)
from .dispatch import dispatch_command


def main() -> None:
    """Entry point for the BN-Syn CLI."""
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(dispatch_command(args))


__all__ = [
    "main",
    "_cmd_demo",
    "_cmd_dtcheck",
    "_cmd_run_experiment",
    "_cmd_sleep_stack",
]
