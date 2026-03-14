import json
from pathlib import Path

from aoc.cli import main


def test_auditor_reliability_trace_emitted(tmp_path: Path, monkeypatch: object) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
 task_prompt: demo
 acceptance_criteria: [ok]
 normalized_constraints:
   target_score: 0.55
   functional_threshold: 0.55
   initial_score: 0.05
   initial_step_size: 0.1
   required_artifact_keys: [status, score, iteration, task]
 innovation_band: {min_delta: 0.0, max_delta: 0.6}
 evaluator_config: {deterministic: true}
 invariants: [artifact_is_json, non_negative_score]
 max_iterations: 8
 coherence_threshold: 0.9
 output_dir: out
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["aoc", "run", "--config", str(cfg)])
    assert main() == 0
    payload = json.loads((tmp_path / "out" / "auditor_reliability_trace.json").read_text())
    assert "records" in payload
    assert len(payload["records"]) >= 1
