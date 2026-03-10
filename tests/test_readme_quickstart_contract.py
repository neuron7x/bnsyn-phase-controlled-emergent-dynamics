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
        "bnsyn run --profile canonical --plot --export-proof",
        "git clone https://github.com/neuron7x/bnsyn-phase-controlled-emergent-dynamics.git",
        "artifacts/canonical_run/emergence_plot.png",
        "artifacts/canonical_run/summary_metrics.json",
        "artifacts/canonical_run/run_manifest.json",
        "artifacts/canonical_run/criticality_report.json",
        "artifacts/canonical_run/avalanche_report.json",
        "artifacts/canonical_run/phase_space_report.json",
        "artifacts/canonical_run/proof_report.json",
    }
    for command in expected:
        assert command in readme

    smoke_commands = _quickstart_smoke_commands()
    assert "python -m scripts.check_quickstart_consistency" in smoke_commands
    assert "python -m pip install -e ." in smoke_commands
    assert "bnsyn --help" in smoke_commands
    assert "bnsyn run --help" in smoke_commands
    assert any(cmd.startswith("bnsyn run --profile canonical --plot --export-proof --output artifacts/canonical_run | python -c") for cmd in smoke_commands)
