import json
import os
import subprocess
import sys
from pathlib import Path


REQUIRED = [
    "final_artifact.json",
    "zeropoint.json",
    "run_summary.json",
    "sigma_trace.json",
    "delta_trace.json",
    "audit_trace.json",
    "auditor_reliability_trace.json",
    "termination_verdict.json",
]


def test_controller_e2e_deterministic(tmp_path: Path) -> None:
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

    cmd = [sys.executable, "-m", "aoc.cli", "run", "--config", str(cfg)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    subprocess.run(cmd, cwd=tmp_path, check=True, env=env)
    verdict1 = json.loads((tmp_path / "out" / "termination_verdict.json").read_text())
    files = {p.name for p in (tmp_path / "out").iterdir()}
    for name in REQUIRED:
        assert name in files
    assert (tmp_path / "out" / "evidence_bundle").is_dir()

    subprocess.run(cmd, cwd=tmp_path, check=True, env=env)
    verdict2 = json.loads((tmp_path / "out" / "termination_verdict.json").read_text())
    assert verdict1 == verdict2
    assert verdict1["status"] in {"PASS", "FAIL", "MAX_ITER", "INCONCLUSIVE"}
