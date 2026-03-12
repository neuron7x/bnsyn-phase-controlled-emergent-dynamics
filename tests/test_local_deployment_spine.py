from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_local_scripts_exist_and_executable() -> None:
    for rel in ("scripts/bootstrap_local_linux.sh", "scripts/run_canonical_local.sh", "scripts/local_doctor.py"):
        path = Path(rel)
        assert path.exists(), f"missing {rel}"


def test_makefile_local_targets_present() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    for target in ("local-bootstrap:", "local-verify:", "local-run:", "local-all:"):
        assert target in makefile


def test_local_doctor_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.local_doctor", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Validate local environment" in result.stdout


def test_bootstrap_changes_to_repo_root() -> None:
    text = Path("scripts/bootstrap_local_linux.sh").read_text(encoding="utf-8")
    assert 'cd "${ROOT_DIR}"' in text


def test_run_script_supports_smoke_and_output_args() -> None:
    text = Path("scripts/run_canonical_local.sh").read_text(encoding="utf-8")
    assert "--smoke" in text
    assert "--output" in text
