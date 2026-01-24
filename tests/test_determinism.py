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
        text = path.read_text()
        for line_no, line in enumerate(text.splitlines(), start=1):
            if "np.random." in line:
                if "np.random.default_rng" in line or "np.random.Generator" in line:
                    continue
                offenders.append(f"{path.relative_to(root)}:{line_no}:{line.strip()}")
    assert offenders == []
