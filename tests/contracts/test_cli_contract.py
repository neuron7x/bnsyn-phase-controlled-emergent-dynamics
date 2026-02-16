from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(Path("src").resolve())
    env["PYTHONPATH"] = f"{src}{os.pathsep}{env.get('PYTHONPATH', '')}" if env.get("PYTHONPATH") else src
    return env


def test_cli_help_and_subcommands_match_contract() -> None:
    contract = json.loads(Path("contracts/cli_contract.v1.json").read_text(encoding="utf-8"))
    proc = subprocess.run(
        [sys.executable, "-m", "bnsyn.cli", "--help"],
        check=True,
        capture_output=True,
        text=True,
        env=_env(),
    )
    text = proc.stdout
    for section in contract["help_sections"]:
        assert section in text
    for sub in contract["subcommands"]:
        assert sub in text


def test_invalid_subcommand_contract() -> None:
    contract = json.loads(Path("contracts/cli_contract.v1.json").read_text(encoding="utf-8"))
    invalid = contract["error_invariants"]["invalid_subcommand"]
    proc = subprocess.run(
        [sys.executable, "-m", "bnsyn.cli", "not-a-cmd"],
        capture_output=True,
        text=True,
        env=_env(),
    )
    assert proc.returncode == invalid["exit_code"]
    assert invalid["stderr_contains"] in proc.stderr
