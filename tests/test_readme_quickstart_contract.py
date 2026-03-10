from __future__ import annotations

from pathlib import Path


def _quickstart_smoke_commands() -> set[str]:
    makefile = Path("Makefile").read_text(encoding="utf-8").splitlines()
    in_target = False
    commands: set[str] = set()
    for line in makefile:
        if line.startswith("quickstart-smoke:"):
            in_target = True
            continue
        if in_target and line and not line.startswith("\t"):
            break
        if in_target and line.startswith("\t"):
            commands.add(line.strip())
    return commands


def test_readme_quickstart_contract_matches_make_target() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    expected = {
        "make quickstart-smoke",
        "python -m pip install -e .",
        "python -m bnsyn --help",
        "bnsyn plot",
        "git clone https://github.com/neuron7x/bnsyn-phase-controlled-emergent-dynamics.git",
        "artifacts/canonical_plot/emergence_plot.png",
        "artifacts/canonical_plot/summary_metrics.json",
        "artifacts/canonical_plot/run_manifest.json",
    }
    for command in expected:
        assert command in readme

    smoke_commands = _quickstart_smoke_commands()
    assert "python -m scripts.check_quickstart_consistency" in smoke_commands
    assert "python -m pip install -e ." in smoke_commands
    assert "bnsyn --help" in smoke_commands
    assert "bnsyn plot --help" in smoke_commands
    assert any(cmd.startswith("bnsyn plot --no-plot | python -c") for cmd in smoke_commands)
