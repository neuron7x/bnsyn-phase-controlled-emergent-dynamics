from __future__ import annotations

import importlib
import inspect
import json
import re
import sys
from pathlib import Path

PUBLIC_API: dict[str, tuple[str, ...]] = {
    "bnsyn": ("__version__",),
    "bnsyn.cli": ("main",),
    "bnsyn.sim.network": ("Network", "NetworkParams", "run_simulation"),
    "bnsyn.neuron.adex": (
        "AdExState",
        "adex_step",
        "adex_step_adaptive",
        "adex_step_with_error_tracking",
    ),
    "bnsyn.synapse.conductance": ("ConductanceState", "ConductanceSynapses", "nmda_mg_block"),
}


def _load_version() -> str:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"', pyproject, flags=re.MULTILINE)
    if not match:
        raise ValueError("Unable to parse project version")
    return match.group(1)


def collect_snapshot() -> dict[str, object]:
    src_path = str((Path(__file__).resolve().parents[1] / "src"))
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    modules: dict[str, dict[str, str]] = {}
    for module_name, symbols in PUBLIC_API.items():
        module = importlib.import_module(module_name)
        symbol_map: dict[str, str] = {}
        for symbol in symbols:
            if not hasattr(module, symbol):
                symbol_map[symbol] = "<missing>"
            else:
                value = getattr(module, symbol)
                try:
                    symbol_map[symbol] = str(inspect.signature(value))
                except (TypeError, ValueError):
                    symbol_map[symbol] = "<non-callable>"
        modules[module_name] = symbol_map

    return {"version": _load_version(), "modules": modules}


def main() -> int:
    out = Path("quality/public_api_snapshot.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(collect_snapshot(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
