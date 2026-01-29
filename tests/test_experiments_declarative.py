"""Tests for declarative experiment runner utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pytest

from bnsyn.experiments import declarative
from bnsyn.schemas.experiment import BNSynExperimentConfig


def _minimal_config() -> BNSynExperimentConfig:
    return BNSynExperimentConfig(
        experiment={"name": "quickstart", "version": "v1", "seeds": [1, 2]},
        network={"size": 10},
        simulation={"duration_ms": 1.0, "dt_ms": 0.1},
    )


def test_load_config_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        declarative.load_config(missing)


def test_load_config_invalid_yaml(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("experiment: [unterminated")
    with pytest.raises(ValueError, match="Invalid YAML"):
        declarative.load_config(bad)


def test_load_config_requires_mapping(tmp_path: Path) -> None:
    data = tmp_path / "list.yaml"
    data.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError, match="Config must be a YAML object"):
        declarative.load_config(data)


def test_load_config_validation_error(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        "\n".join(
            [
                "experiment:",
                "  name: bad name",
                "  version: v1",
                "  seeds: [1]",
                "network:",
                "  size: 10",
                "simulation:",
                "  duration_ms: 1.0",
                "  dt_ms: 0.1",
            ]
        )
    )
    with pytest.raises(ValueError, match="Config validation failed"):
        declarative.load_config(invalid)


def test_run_experiment_collects_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _minimal_config()
    calls: list[dict[str, Any]] = []

    def fake_run_simulation(*, steps: int, dt_ms: float, seed: int, N: int) -> dict[str, Any]:
        calls.append({"steps": steps, "dt_ms": dt_ms, "seed": seed, "N": N})
        return {"seed": seed, "steps": steps}

    monkeypatch.setattr(declarative, "run_simulation", fake_run_simulation)

    results = declarative.run_experiment(config)
    assert results["config"]["name"] == "quickstart"
    assert len(results["runs"]) == len(config.experiment.seeds)

    expected_steps = int(config.simulation.duration_ms / config.simulation.dt_ms)
    assert calls == [
        {"steps": expected_steps, "dt_ms": 0.1, "seed": 1, "N": 10},
        {"steps": expected_steps, "dt_ms": 0.1, "seed": 2, "N": 10},
    ]


def test_run_from_yaml_writes_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "experiment:",
                "  name: quickstart",
                "  version: v1",
                "  seeds: [1, 2]",
                "network:",
                "  size: 10",
                "simulation:",
                "  duration_ms: 1.0",
                "  dt_ms: 0.1",
            ]
        )
    )
    output_file = tmp_path / "results" / "out.json"
    stub_result = {"runs": [{"seed": 1, "metrics": {"sigma": 1.0}}]}

    def fake_run_experiment(_: BNSynExperimentConfig) -> dict[str, Any]:
        return stub_result

    monkeypatch.setattr(declarative, "run_experiment", fake_run_experiment)

    declarative.run_from_yaml(config_file, output_file)
    payload = json.loads(output_file.read_text())
    assert payload == stub_result


def test_run_from_yaml_prints_when_no_output(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    config = _minimal_config()
    stub_result = {"runs": [{"seed": 1, "metrics": {"sigma": 1.0}}]}

    monkeypatch.setattr(declarative, "load_config", lambda _: config)
    monkeypatch.setattr(declarative, "run_experiment", lambda _: stub_result)

    declarative.run_from_yaml("config.yaml")
    captured = capsys.readouterr().out
    assert "Config validated" in captured
    assert json.dumps(stub_result, indent=2, sort_keys=True) in captured
