import json
import os
import subprocess
import sys
from pathlib import Path


def test_cli_end_to_end(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
task_prompt: cli demo
acceptance_criteria: [ok]
normalized_constraints:
  target_score: 0.40
  functional_threshold: 0.40
  initial_score: 0.10
  initial_step_size: 0.10
  required_artifact_keys: [status, score, iteration, task]
innovation_band: {min_delta: 0.0, max_delta: 0.7}
delta_weights: {semantic: 0.4, structural: 0.3, functional: 0.3}
evaluator_config: {deterministic: true}
invariants: [artifact_is_json, non_negative_score]
artifact_expectations: [status, score, iteration, task]
max_iterations: 5
coherence_threshold: 0.9
output_dir: out
""",
        encoding="utf-8",
    )

    cmd = [sys.executable, "-m", "aoc.cli", "run", "--config", str(cfg)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    completed = subprocess.run(cmd, cwd=tmp_path, check=True, capture_output=True, text=True, env=env)
    parsed = json.loads(completed.stdout)
    assert parsed["status"] in {"PASS", "FAIL", "MAX_ITER", "INCONCLUSIVE"}
