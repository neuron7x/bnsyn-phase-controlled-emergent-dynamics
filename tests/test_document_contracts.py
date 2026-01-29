from __future__ import annotations

import subprocess
from pathlib import Path


def test_document_contracts_are_valid() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "validate_document_contracts.py"
    subprocess.run(["python", str(script)], check=True)
