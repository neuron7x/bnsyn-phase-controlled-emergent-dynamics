"""Runner integration tests for pytest diagnostics exit semantics and publication."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from bnsyn.qa.pytest_failure_diagnostics import PublicationOptions, generate_diagnostics, publish_ci_outputs

SCHEMA = Path("schemas/pytest-failure-diagnostics.schema.json")


def _run_runner(tmp_path: Path, test_code: str, markers: str = "", extra_pytest_args: list[str] | None = None) -> subprocess.CompletedProcess[str]:
    test_file = tmp_path / "synthetic_test.py"
    test_file.write_text(test_code, encoding="utf-8")

    junit = tmp_path / "junit.xml"
    log = tmp_path / "pytest.log"
    out_json = tmp_path / "diag.json"
    out_md = tmp_path / "diag.md"

    cmd = [
        sys.executable,
        "-m",
        "scripts.run_pytest_with_diagnostics",
        "--markers",
        markers,
        "--junit",
        str(junit),
        "--log",
        str(log),
        "--output-json",
        str(out_json),
        "--output-md",
        str(out_md),
        "--schema",
        str(SCHEMA),
        str(test_file),
    ]
    if extra_pytest_args:
        cmd.extend(extra_pytest_args)

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{Path.cwd() / 'src'}:{env.get('PYTHONPATH', '')}"
    return subprocess.run(cmd, text=True, capture_output=True, check=False, env=env)


def test_runner_preserves_zero_exit_code_and_emits_artifacts(tmp_path: Path) -> None:
    result = _run_runner(tmp_path, "def test_ok():\n    assert True\n")
    assert result.returncode == 0
    assert (tmp_path / "diag.json").exists()
    assert (tmp_path / "diag.md").exists()


def test_runner_preserves_nonzero_exit_code_and_emits_artifacts(tmp_path: Path) -> None:
    result = _run_runner(tmp_path, "def test_fail():\n    assert False\n")
    assert result.returncode == 1
    payload = json.loads((tmp_path / "diag.json").read_text(encoding="utf-8"))
    assert payload["pytest_exit_code"] == 1
    assert payload["status"] == "failures_detected"


def test_runner_invalid_marker_preserves_error_semantics(tmp_path: Path) -> None:
    result = _run_runner(tmp_path, "def test_ok():\n    assert True\n", markers="(")
    assert result.returncode != 0
    payload = json.loads((tmp_path / "diag.json").read_text(encoding="utf-8"))
    assert payload["pytest_exit_code"] == result.returncode


def test_publication_helpers_are_bounded_and_deterministic(tmp_path: Path) -> None:
    junit = tmp_path / "junit.xml"
    junit.write_text(
        """
<testsuite name="suite" tests="3" failures="3" errors="0" skipped="0">
  <testcase classname="t" name="test_a" file="a.py"><failure message="m">tb</failure></testcase>
  <testcase classname="t" name="test_b" file="b.py"><failure message="m">tb</failure></testcase>
  <testcase classname="t" name="test_c" file="c.py"><failure message="m">tb</failure></testcase>
</testsuite>
""".strip(),
        encoding="utf-8",
    )
    out_json = tmp_path / "diag.json"
    out_md = tmp_path / "diag.md"
    payload = generate_diagnostics(
        junit_xml=junit,
        output_json=out_json,
        output_md=out_md,
        pytest_exit_code=1,
        schema_path=SCHEMA,
    )

    annotations = tmp_path / "ann.txt"
    summary = tmp_path / "summary.md"
    meta = publish_ci_outputs(
        payload,
        PublicationOptions(annotations_file=annotations, max_annotations=2, github_step_summary=summary),
    )
    assert meta["annotations_emitted"] == 2
    annotation_lines = annotations.read_text(encoding="utf-8").strip().splitlines()
    assert len(annotation_lines) == 2
    assert annotation_lines == sorted(annotation_lines)
    assert "status:" in summary.read_text(encoding="utf-8")
