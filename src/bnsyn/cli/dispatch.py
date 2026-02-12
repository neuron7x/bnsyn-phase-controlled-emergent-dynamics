"""Command routing for parsed BN-Syn CLI arguments."""

from __future__ import annotations

import argparse
from collections.abc import Callable

from .commands import cmd_demo, cmd_dtcheck, cmd_run_experiment, cmd_sleep_stack

CommandHandler = Callable[[argparse.Namespace], int]

COMMAND_HANDLERS: dict[str, CommandHandler] = {
    "demo": cmd_demo,
    "run": cmd_run_experiment,
    "dtcheck": cmd_dtcheck,
    "sleep-stack": cmd_sleep_stack,
}


def dispatch_command(args: argparse.Namespace) -> int:
    """Dispatch parsed arguments to the selected command handler."""
    command = getattr(args, "cmd")
    handler = COMMAND_HANDLERS[command]
    return int(handler(args))
