"""Version discovery helpers for CLI provenance metadata."""

from __future__ import annotations

import importlib.metadata
import tomllib
import warnings
from pathlib import Path


def get_package_version() -> str:
    """Return installed package version with a filesystem fallback."""
    version: str | None = None
    try:
        version = importlib.metadata.version("bnsyn")
    except importlib.metadata.PackageNotFoundError:
        version = None
    except Exception as exc:
        warnings.warn(f"Failed to read package version: {exc}", stacklevel=2)
        version = None

    if version:
        return version

    pyproject_path = Path(__file__).resolve().parents[3] / "pyproject.toml"
    if pyproject_path.exists():
        try:
            data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            return "unknown"
        version = data.get("project", {}).get("version")
        if isinstance(version, str) and version:
            return version

    return "unknown"
