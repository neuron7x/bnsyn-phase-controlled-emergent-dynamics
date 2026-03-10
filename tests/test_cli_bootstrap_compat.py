from __future__ import annotations

from types import SimpleNamespace

import pytest

from bnsyn import cli


def test_cmd_run_experiment_compat_without_optional_attrs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str | None]] = []

    def fake_run_from_yaml(config: str, output: str | None) -> None:
        calls.append((config, output))

    monkeypatch.setitem(__import__("sys").modules, "bnsyn.experiments.declarative", SimpleNamespace(run_from_yaml=fake_run_from_yaml))

    args = SimpleNamespace(config="examples/configs/quickstart.yaml", output=None)
    rc = cli._cmd_run_experiment(args)

    assert rc == 0
    assert calls == [("examples/configs/quickstart.yaml", None)]


def test_cmd_run_experiment_routes_profile_canonical(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_run_from_yaml(config: str, output: str | None) -> None:
        del output
        seen.append(config)

    monkeypatch.setitem(__import__("sys").modules, "bnsyn.experiments.declarative", SimpleNamespace(run_from_yaml=fake_run_from_yaml))

    args = SimpleNamespace(config=None, profile="canonical", output=None)
    rc = cli._cmd_run_experiment(args)

    assert rc == 0
    assert seen == ["configs/canonical_profile.yaml"]


def test_cmd_run_experiment_missing_config_returns_2(capsys: pytest.CaptureFixture[str]) -> None:
    args = SimpleNamespace()
    rc = cli._cmd_run_experiment(args)
    captured = capsys.readouterr()

    assert rc == 2
    assert "provide CONFIG or --profile canonical" in captured.err


def test_cmd_run_experiment_prints_reserved_flag_notices(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_run_from_yaml(config: str, output: str | None) -> None:
        del config, output

    monkeypatch.setitem(__import__("sys").modules, "bnsyn.experiments.declarative", SimpleNamespace(run_from_yaml=fake_run_from_yaml))

    args = SimpleNamespace(config="examples/configs/quickstart.yaml", output=None, plot=True, export_proof=True)
    rc = cli._cmd_run_experiment(args)
    captured = capsys.readouterr()

    assert rc == 0
    assert "--plot is part of the reserved canonical interface" in captured.out
    assert "--export-proof is part of the reserved canonical interface" in captured.out
