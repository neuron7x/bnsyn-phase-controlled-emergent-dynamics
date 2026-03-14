import json
import os
import subprocess
import sys
from pathlib import Path


def test_evidence_bundle_emitted_on_failure_path(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg_failure.yaml"
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
delta_weights: {semantic: 0.4, structural: 0.3, functional: 0.3}
evaluator_config: {deterministic: true}
invariants: [artifact_is_json, unknown_invariant]
artifact_expectations: [status, score, iteration, task]
max_iterations: 3
coherence_threshold: 0.9
output_dir: out
""",
        encoding="utf-8",
    )
    cmd = [sys.executable, "-m", "aoc.cli", "run", "--config", str(cfg)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    subprocess.run(cmd, cwd=tmp_path, check=True, env=env)

    verdict = json.loads((tmp_path / "out" / "termination_verdict.json").read_text())
    assert verdict["status"] in {"FAIL", "MAX_ITER", "INCONCLUSIVE"}
    assert (tmp_path / "out" / "evidence_bundle" / "termination_verdict.json").exists()
    assert (tmp_path / "out" / "evidence_bundle" / "zeropoint.json").exists()
