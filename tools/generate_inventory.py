from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(".")
    paths = ["src", "tests", "docs", ".github"]
    counts = {path: sum(1 for item in (root / path).rglob("*") if item.is_file()) for path in paths}
    workflows = sorted((root / ".github" / "workflows").glob("*.yml"))
    workflow_names = [path.name for path in workflows]
    detected_tools = [
        name
        for name in ["pyproject.toml", "requirements-lock.txt", "docker-compose.yml", "Makefile"]
        if (root / name).exists()
    ]
    inventory = {
        "_generated_by": "python tools/generate_inventory.py",
        "paths": paths,
        "counts": counts,
        "workflows": workflow_names,
        "detected_tools": detected_tools,
    }
    root.joinpath("INVENTORY.json").write_text(json.dumps(inventory, indent=2) + "\n")


if __name__ == "__main__":
    main()
