"""Contract tests for mutation run pipeline crash/survivor handling."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_run_pipeline_accepts_survivor_exit_code(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import scripts.run_mutation_pipeline as run_mutation_pipeline

    calls: list[tuple[str, ...]] = []

    def fake_run(args: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(tuple(args))
        if args[:2] == ["mutmut", "run"]:
            return SimpleNamespace(returncode=1, stdout="run", stderr="survivors")
        if args[:2] == ["mutmut", "results"]:
            return SimpleNamespace(returncode=0, stdout="results", stderr="")
        if args[:3] == ["mutmut", "show", "--status"]:
            return SimpleNamespace(returncode=0, stdout="1\n", stderr="")
        raise AssertionError(args)

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.chdir(tmp_path)

    assert run_mutation_pipeline.main([]) == 0
    assert (tmp_path / "mutation_results.txt").read_text(encoding="utf-8") == "results"
    assert (tmp_path / "survived_mutants.txt").read_text(encoding="utf-8") == "1\n"
    assert any(call[:2] == ("mutmut", "results") for call in calls)


def test_run_pipeline_fails_when_results_unavailable(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import scripts.run_mutation_pipeline as run_mutation_pipeline

    def fake_run(args: list[str], **_kwargs: object) -> SimpleNamespace:
        if args[:2] == ["mutmut", "run"]:
            return SimpleNamespace(returncode=2, stdout="run", stderr="crash")
        if args[:2] == ["mutmut", "results"]:
            return SimpleNamespace(returncode=2, stdout="", stderr="no cache")
        raise AssertionError(args)

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.chdir(tmp_path)

    assert run_mutation_pipeline.main([]) == 1
    assert not (tmp_path / "mutation_results.txt").exists()
