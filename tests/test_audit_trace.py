import json
from pathlib import Path

from aoc.cli import main


def test_reliability_trace_collected(tmp_path: Path, monkeypatch: object) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(Path("examples/basic_task.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["aoc", "run", "--config", str(cfg)])
    assert main() == 0
    trace = json.loads((tmp_path / "aoc_output" / "auditor_reliability_trace.json").read_text())
    assert "history" in trace
    assert len(trace["history"]) > 0
