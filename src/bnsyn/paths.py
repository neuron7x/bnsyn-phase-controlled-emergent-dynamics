from __future__ import annotations

import tempfile
from importlib import resources
from pathlib import Path


def _candidate_roots() -> list[Path]:
    module_root = Path(__file__).resolve()
    cwd = Path.cwd().resolve()
    candidates: list[Path] = []
    candidates.extend(module_root.parents)
    candidates.append(cwd)
    candidates.extend(cwd.parents)
    return candidates


def repo_file(relative_path: str) -> Path:
    rel = Path(relative_path)
    for root in _candidate_roots():
        marker = root / "pyproject.toml"
        candidate = root / rel
        if marker.exists() and candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to locate required repository file: {relative_path}")


def package_file(relative_path: str) -> Path:
    traversable = resources.files("bnsyn").joinpath("resources").joinpath(relative_path)
    if not traversable.is_file():
        raise FileNotFoundError(f"Packaged resource missing: {relative_path}")
    cache_root = Path(tempfile.gettempdir()) / "bnsyn_resources"
    target = cache_root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    data = traversable.read_bytes()
    if not target.exists() or target.read_bytes() != data:
        target.write_bytes(data)
    return target


def runtime_file(relative_path: str) -> Path:
    """Resolve runtime resources from packaged data first, then repository checkout."""
    try:
        return package_file(relative_path)
    except FileNotFoundError:
        return repo_file(relative_path)
