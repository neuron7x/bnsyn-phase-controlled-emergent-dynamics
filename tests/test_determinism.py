import json
import os
import subprocess
import sys
from pathlib import Path

from bnsyn.sim.network import run_simulation


def test_determinism_same_seed_same_metrics() -> None:
    m1 = run_simulation(steps=500, dt_ms=0.1, seed=42, N=80)
    m2 = run_simulation(steps=500, dt_ms=0.1, seed=42, N=80)
    assert m1 == m2


def test_no_global_numpy_rng_usage() -> None:
    root = Path(__file__).resolve().parents[1]
    src_root = root / "src" / "bnsyn"
    offenders: list[str] = []
    for path in src_root.rglob("*.py"):
        rel_path = path.relative_to(root)
        text = path.read_text()
        for line_no, line in enumerate(text.splitlines(), start=1):
            if "np.random." in line:
                if "np.random.Generator" in line:
                    continue
                if rel_path.as_posix() == "src/bnsyn/rng.py":
                    continue
                offenders.append(f"{rel_path}:{line_no}:{line.strip()}")
            if "random." in line or line.strip().startswith("import random"):
                offenders.append(f"{rel_path}:{line_no}:{line.strip()}")
    assert offenders == []


def test_determinism_across_pythonhashseed_for_summary_output() -> None:
    cmd = [
        sys.executable,
        "-c",
        (
            "import json; "
            "from bnsyn.sim.network import run_simulation; "
            "print(json.dumps(run_simulation(steps=200, dt_ms=0.1, seed=42, N=64), sort_keys=True))"
        ),
    ]
    env0 = os.environ.copy()
    env1 = os.environ.copy()
    env0["PYTHONHASHSEED"] = "0"
    env1["PYTHONHASHSEED"] = "123"

    repo_root = Path(__file__).resolve().parents[1]
    src_path = str(repo_root / "src")
    env0["PYTHONPATH"] = src_path + os.pathsep + env0.get("PYTHONPATH", "")
    env1["PYTHONPATH"] = src_path + os.pathsep + env1.get("PYTHONPATH", "")

    out0 = subprocess.check_output(cmd, text=True, env=env0).strip()
    out1 = subprocess.check_output(cmd, text=True, env=env1).strip()

    assert json.loads(out0) == json.loads(out1)
