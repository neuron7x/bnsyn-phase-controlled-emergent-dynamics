import json
from pathlib import Path

from aoc.contracts import load_task_contract
from aoc.controller import AOCController


REQUIRED = {
    "final_artifact.md",
    "zeropoint.json",
    "run_summary.json",
    "sigma_trace.json",
    "delta_trace.json",
    "audit_trace.json",
    "auditor_reliability_trace.json",
    "termination_verdict.json",
}


def test_full_controller_run_and_determinism(tmp_path: Path) -> None:
    payload = json.loads(json.dumps(__import__("yaml").safe_load(Path("examples/basic_task.yaml").read_text())))
    payload["output"]["artifact_filename"] = "final_artifact.md"
    c = load_task_contract(payload)

    out = tmp_path / "run"
    v1 = AOCController(c, out).run()
    files = {p.name for p in out.iterdir()}
    assert REQUIRED.issubset(files)
    assert (out / "evidence_bundle").is_dir()

    v2 = AOCController(c, out).run()
    assert v1 == v2
