"""CLI entrypoint for BN-Syn benchmarks."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    from scripts.run_benchmarks import main as run_main

    return run_main()
