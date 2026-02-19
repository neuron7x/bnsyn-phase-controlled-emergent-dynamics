"""Proof helpers for GHTPO artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_hashes(paths: list[Path], out_file: Path) -> None:
    rows = [{"path": str(path).replace('\\\\', '/'), "sha256": sha256_file(path)} for path in sorted(paths)]
    out_file.write_text(json.dumps({"files": rows}, indent=2) + "\n", encoding="utf-8")
