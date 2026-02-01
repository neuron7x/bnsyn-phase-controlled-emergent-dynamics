from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT_DIR / "manifest.json"
DEFAULT_ROLE = "template_asset"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_ref() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def _iter_files() -> list[Path]:
    files: list[Path] = []
    for path in sorted(ROOT_DIR.rglob("*")):
        if path.is_dir():
            continue
        if path.name == "manifest.json":
            continue
        if "__pycache__" in path.parts:
            continue
        if path.suffix == ".pyc":
            continue
        files.append(path)
    return files


def _role_for(relpath: str) -> str:
    if relpath == "README.md":
        return "contract"
    if relpath.startswith("tools/"):
        return "tool"
    return DEFAULT_ROLE


def _build_manifest(seed: int) -> dict:
    random.seed(seed)
    script_rel = Path("tools") / "update_manifest.py"
    entries = []
    for path in _iter_files():
        relpath = path.relative_to(ROOT_DIR).as_posix()
        entries.append(
            {
                "path": relpath,
                "sha256": _sha256(path),
                "role": _role_for(relpath),
                "reproducibility_command_or_na": f"python {script_rel.as_posix()}",
            }
        )
    return {
        "schema_version": "1.0",
        "generated_by": script_rel.as_posix(),
        "provenance": {
            "repo_ref": _git_ref(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed": seed,
        },
        "entries": entries,
    }


def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _check_manifest(seed: int) -> int:
    existing = _load_manifest()
    if not existing:
        print("manifest.json is missing", file=sys.stderr)
        return 1
    expected = _build_manifest(seed)
    expected_entries = expected["entries"]
    existing_entries = existing.get("entries", [])
    if expected_entries != existing_entries:
        print("manifest entries are out of date", file=sys.stderr)
        return 1
    if existing.get("schema_version") != expected.get("schema_version"):
        print("schema_version mismatch", file=sys.stderr)
        return 1
    if existing.get("generated_by") != expected.get("generated_by"):
        print("generated_by mismatch", file=sys.stderr)
        return 1
    if existing.get("provenance", {}).get("seed") != seed:
        print("seed mismatch", file=sys.stderr)
        return 1
    return 0


def _write_manifest(manifest: dict) -> None:
    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Update or validate manifest.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        return _check_manifest(args.seed)

    manifest = _build_manifest(args.seed)
    _write_manifest(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
