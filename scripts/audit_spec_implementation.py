from __future__ import annotations

import pathlib
import re
import sys
from typing import Iterable

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "docs" / "SPEC.md"
MAP_PATH = ROOT / "docs" / "spec_to_code.yml"

COMPONENT_RE = re.compile(r"^##\s+(P\d-\d+):\s*(.+)$")


def _read_spec_components(lines: Iterable[str]) -> list[tuple[str, str]]:
    components: list[tuple[str, str]] = []
    for line in lines:
        match = COMPONENT_RE.match(line.strip())
        if match:
            components.append((match.group(1), match.group(2).strip()))
    return components


def _has_validation_marker(text: str) -> bool:
    return "pytest.mark.validation" in text or "pytestmark = pytest.mark.validation" in text


def main() -> int:
    if not SPEC_PATH.exists():
        print(f"Missing SPEC: {SPEC_PATH}")
        return 1
    if not MAP_PATH.exists():
        print(f"Missing spec_to_code map: {MAP_PATH}")
        return 1

    spec_components = _read_spec_components(SPEC_PATH.read_text().splitlines())
    if len(spec_components) != 12:
        print(f"Expected 12 spec components, found {len(spec_components)}")
        for cid, name in spec_components:
            print(f"  - {cid}: {name}")
        return 1

    mapping = yaml.safe_load(MAP_PATH.read_text()) or {}
    components = mapping.get("components", {})
    if not isinstance(components, dict):
        print("spec_to_code.yml missing 'components' mapping")
        return 1

    errors: list[str] = []
    for comp_id, comp_name in spec_components:
        entry = components.get(comp_id)
        if entry is None:
            errors.append(f"Missing mapping for {comp_id} ({comp_name})")
            continue
        impl_paths = entry.get("implementation_paths") or []
        ver_paths = entry.get("verification_paths") or []
        if not impl_paths:
            errors.append(f"{comp_id} missing implementation_paths")
        if not ver_paths:
            errors.append(f"{comp_id} missing verification_paths")
        for rel in impl_paths:
            path = ROOT / rel
            if not path.exists():
                errors.append(f"{comp_id} implementation missing: {rel}")
        for rel in ver_paths:
            path = ROOT / rel
            if not path.exists():
                errors.append(f"{comp_id} verification missing: {rel}")
                continue
            text = path.read_text()
            if "tests/validation" in rel.replace("\\", "/"):
                if not _has_validation_marker(text):
                    errors.append(f"{comp_id} validation test missing pytest.mark.validation: {rel}")
            else:
                if _has_validation_marker(text):
                    errors.append(f"{comp_id} smoke test incorrectly marked validation: {rel}")

    if errors:
        print("Spec implementation audit failed:")
        for err in errors:
            print(f" - {err}")
        return 1

    print("Spec implementation audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
