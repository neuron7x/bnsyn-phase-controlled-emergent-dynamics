from __future__ import annotations

from bnsyn import phase_atlas, run, sleep_stack


def test_api_run_returns_expected_keys() -> None:
    out = run({"steps": 20, "dt_ms": 0.1, "seed": 7, "N": 24})
    assert set(out) == {"sigma_mean", "rate_mean_hz", "sigma_std", "rate_std"}


def test_api_phase_atlas_contract() -> None:
    atlas = phase_atlas(seed=123)
    assert atlas["schema_version"] == "1.0.0"
    assert atlas["seed"] == 123
    assert len(atlas["records"]) >= 1


def test_api_sleep_stack_contract() -> None:
    out = sleep_stack({"seed": 9, "N": 32, "out": "results/demo"})
    assert out["seed"] == 9
    assert out["N"] == 32
    assert out["out"] == "results/demo"
