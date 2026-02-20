from __future__ import annotations

import json

from scripts import titan9_l3_validation


def test_activation_command_contains_target() -> None:
    cmd = titan9_l3_validation.activation_command("bnsyn")
    assert "Activate TITAN-9 L3" in cmd
    assert "Target: bnsyn" in cmd


def test_run_validation_returns_pass_for_benchmark_meeting_metrics() -> None:
    metrics = {pillar: 1.0 for pillar in titan9_l3_validation.PILLARS}
    benchmarks = {pillar: 1.0 for pillar in titan9_l3_validation.PILLARS}

    result = titan9_l3_validation.run_validation(metrics=metrics, benchmarks=benchmarks)

    assert result.binary == "PASS"
    assert result.status == "STABLE"
    assert len(result.rows) == 9


def test_main_json_output_with_telemetry_file(tmp_path, capsys, monkeypatch) -> None:
    payload = {
        "metrics": {pillar: 0.95 for pillar in titan9_l3_validation.PILLARS},
        "benchmarks": {pillar: 1.0 for pillar in titan9_l3_validation.PILLARS},
    }
    telemetry = tmp_path / "telemetry.json"
    telemetry.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "titan9_l3_validation",
            "--telemetry",
            str(telemetry),
            "--format",
            "json",
        ],
    )

    exit_code = titan9_l3_validation.main()
    captured = capsys.readouterr().out
    rendered = json.loads(captured)

    assert exit_code == 0
    assert rendered["status"] == "DEGRADED"
    assert rendered["binary"] == "FAIL"
