#!/usr/bin/env python3
"""Check public API docstring coverage for the bnsyn package."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from dataclasses import dataclass
from pathlib import Path

MIN_DOCSTRING_LEN = 30
SKIP_MODULES = {
    "bnsyn.production.jax_backend",
}


@dataclass(frozen=True)
class MissingDoc:
    symbol: str
    reason: str


def iter_modules(package_name: str) -> list[str]:
    package = importlib.import_module(package_name)
    modules = [package.__name__]
    if hasattr(package, "__path__"):
        modules.extend([m.name for m in pkgutil.walk_packages(package.__path__, package.__name__ + ".")])
    return modules


def docstring_ok(doc: str | None) -> bool:
    if doc is None:
        return False
    return len(doc.strip()) >= MIN_DOCSTRING_LEN


def check_module(module_name: str) -> list[MissingDoc]:
    if module_name in SKIP_MODULES:
        return []
    module = importlib.import_module(module_name)
    missing: list[MissingDoc] = []

    if not docstring_ok(module.__doc__):
        missing.append(MissingDoc(symbol=f"{module_name} (module)", reason="missing module docstring"))

    for name, member in inspect.getmembers(module):
        if name.startswith("_"):
            continue
        if inspect.isfunction(member) or inspect.isclass(member):
            if not docstring_ok(member.__doc__):
                missing.append(MissingDoc(symbol=f"{module_name}.{name}", reason="missing docstring"))

    return missing


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if not (root / "src" / "bnsyn").exists():
        raise SystemExit("Expected src/bnsyn package not found")

    modules = iter_modules("bnsyn")
    missing: list[MissingDoc] = []

    for module_name in modules:
        missing.extend(check_module(module_name))

    print(f"[docstrings] Modules scanned: {len(modules)}")
    print(f"[docstrings] Minimum length: {MIN_DOCSTRING_LEN}")
    print(f"[docstrings] Missing docstrings: {len(missing)}")

    if missing:
        for entry in missing:
            print(f"[docstrings] MISSING: {entry.symbol} ({entry.reason})")
        return 1

    print("[docstrings] OK: all public symbols have docstrings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
