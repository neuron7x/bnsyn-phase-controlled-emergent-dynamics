from __future__ import annotations

import json
from pathlib import Path

from sse_policy_load import load_and_validate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "sse_sdo" / "01_scope"


def build_inventory() -> tuple[dict[str, object], dict[str, object]]:
    policy = load_and_validate()
    paths: list[str] = policy["scope"]["subsystem_paths"]
    dep_graph = {"nodes": sorted(paths), "edges": []}
    interface_registry = {
        "interfaces": [
            {"path": p, "exists": (ROOT / p).exists()} for p in sorted(paths)
        ]
    }
    return dep_graph, interface_registry


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    dep_graph, interface_registry = build_inventory()
    (OUT / "DEP_GRAPH.json").write_text(json.dumps(dep_graph, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "INTERFACE_REGISTRY.json").write_text(
        json.dumps(interface_registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
