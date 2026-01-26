# TEST POLICY — BN-Syn

## Overview

This document defines the test architecture for BN-Syn, including test tiering, runtime budgets, and enforcement mechanisms. The policy ensures fast iteration during development while maintaining thorough validation coverage.

---

## Test Tiers

### Smoke Tests (Critical Path)

**Purpose**: Fast verification of core functionality and integration points.

**Characteristics**:
- Runtime: <30 seconds total
- Deterministic: No statistical sampling required
- No external dependencies: Must work with minimal install
- Scope: Core algorithms, API contracts, determinism, type safety

**Marker**: `@pytest.mark.smoke` (optional; default for unmarked tests)

**CI Integration**: Run on every PR (required to pass)

**Examples**:
- Neuron dynamics correctness (AdEx equations)
- Synapse conductance calculations
- Plasticity weight updates
- RNG determinism validation
- Optional import isolation
- Network construction and single-step simulation
- Temperature controller state transitions

**Command**:
```bash
pytest -m "not validation" -q
make test
```

### Validation Tests (Statistical & Visual)

**Purpose**: Thorough verification of statistical properties, long-running experiments, and visualization.

**Characteristics**:
- Runtime: 2-5 minutes (statistical experiments with multiple seeds)
- May require optional dependencies (matplotlib for visualization)
- Statistical validation: Multiple random seeds, distributions, avalanche statistics
- Visual outputs: Dashboard rendering, animation generation

**Marker**: `@pytest.mark.validation` (required)

**CI Integration**: Separate workflow (optional for PRs, required for releases)

**Examples**:
- Multi-seed temperature ablation experiments
- Criticality phase transition analysis
- Avalanche size/duration distributions
- Memory consolidation stability across conditions
- Dashboard visualization generation
- Animation frame rendering

**Command**:
```bash
pytest -m validation -q
make test-validation
```

---

## Enforcement

### Pytest Configuration

Defined in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
addopts = "-q --strict-markers"
testpaths = ["tests"]
markers = [
  "smoke: fast critical-path tests",
  "validation: slow statistical/large-N validation tests (excluded from CI by default)",
]
```

**Key Settings**:
- `--strict-markers`: Fail on unknown markers (prevents typos)
- Default run excludes validation (use `-m "not validation"`)

### CI Guard Test

A meta-test ensures validation tests are properly marked:

```python
# tests/test_marker_enforcement.py
def test_validation_tests_are_marked():
    """Ensure all slow tests are marked as validation."""
    # Collect all tests
    all_tests = pytest.main(["--collect-only", "-q", "tests/"])
    validation_tests = pytest.main(["--collect-only", "-q", "-m", "validation", "tests/"])
    
    # Smoke tests should complete quickly
    # (This test itself is a smoke test)
```

**Purpose**: Prevent accidental inclusion of slow tests in smoke suite.

### Makefile Targets

Defined in `Makefile`:

```makefile
test:
    pytest -m "not validation" -v --tb=short

test-determinism:
    pytest tests/test_determinism.py tests/test_properties_determinism.py -v

test-validation:
    pytest -m validation -v
```

**Usage**:
- `make test`: Default developer workflow (smoke only)
- `make test-validation`: Run heavy experiments
- `make test-determinism`: Focused determinism verification

---

## Guidelines for Test Authors

### When to Mark as Validation

A test **MUST** be marked `@pytest.mark.validation` if it:
1. Takes >1 second to run
2. Requires multiple random seeds (>5) for statistical validity
3. Generates visualizations or animations
4. Performs large-scale network simulations (N>1000 neurons)
5. Sweeps parameter spaces with >10 conditions

### When to Keep as Smoke

A test **SHOULD** remain unmarked (smoke) if it:
1. Runs in <1 second
2. Tests a single deterministic case
3. Verifies API contracts or type signatures
4. Checks error handling and edge cases
5. Validates core equations with minimal inputs

### Example: Converting a Slow Test

**Before** (slow test leaking into smoke suite):
```python
def test_temperature_sweep():
    """Test criticality across temperature range."""
    for T in np.linspace(0.1, 2.0, 50):  # 50 conditions!
        network = create_network(T=T)
        # ... expensive simulation ...
```

**After** (properly marked):
```python
@pytest.mark.validation
def test_temperature_sweep():
    """Test criticality across temperature range."""
    for T in np.linspace(0.1, 2.0, 50):
        network = create_network(T=T)
        # ... expensive simulation ...

def test_temperature_effect_smoke():
    """Smoke test: temperature changes criticality (single case)."""
    network_low = create_network(T=0.5)
    network_high = create_network(T=1.5)
    # ... quick single-step check ...
```

**Pattern**: Extract a fast, deterministic smoke test alongside the heavy validation test.

---

## Runtime Budgets

### Smoke Suite
- **Target**: <15 seconds
- **Maximum**: <30 seconds
- **Per-test**: <1 second (typical)

**Monitoring**:
```bash
pytest -m "not validation" --durations=10
```

### Validation Suite
- **Target**: <3 minutes
- **Maximum**: <5 minutes
- **Per-test**: Variable (some may take 30-60 seconds for multi-seed experiments)

**Monitoring**:
```bash
pytest -m validation --durations=10
```

**Action on Violation**:
1. If smoke suite exceeds 30s: Identify slow tests and mark as validation
2. If validation exceeds 5min: Consider test parallelization or sampling reduction

---

## Coverage Policy

**Coverage Target**: 85% line coverage on `src/bnsyn/`

**Measurement**: Smoke tests only (validation excluded from coverage metrics)

```bash
pytest -m "not validation" --cov=src/bnsyn --cov-report=html --cov-fail-under=85
make coverage
```

**Rationale**:
- Validation tests are for statistical properties, not coverage
- Smoke tests must cover all critical paths
- 85% balances thoroughness with pragmatism (some defensive branches are hard to trigger)

**Coverage Gaps Allowed**:
- Unreachable error paths (defensive programming)
- Visualization code (covered by validation, not smoke)
- Optional backend code (JAX, Torch) when not installed

---

## Determinism Requirements

All tests (smoke and validation) **MUST** be deterministic.

**Rules**:
1. Always seed RNG at test start:
   ```python
   from bnsyn.rng import seed_all, RNGPack
   
   def test_something():
       seed_all(42)
       rng = RNGPack.from_seed(42)
       # ... test code ...
   ```

2. Do not rely on wall-clock time or system state
3. Sort dictionaries/sets before assertions when order matters
4. For validation tests with multiple seeds: parametrize explicitly
   ```python
   @pytest.mark.validation
   @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1011])
   def test_multi_seed_experiment(seed):
       rng = RNGPack.from_seed(seed)
       # ... statistical validation ...
   ```

**Enforcement**: Golden hash test locks down deterministic replay (see `tests/test_golden_hash.py`)

---

## Failure Handling

### Flaky Tests

**Definition**: A test that passes/fails intermittently without code changes.

**Policy**: Flaky tests are **not allowed** in either tier.

**Remediation**:
1. Investigate root cause (usually unseeded randomness or timing)
2. Fix determinism issue
3. Re-run 10 times to verify stability
4. If unfixable: Move to validation with explicit `@pytest.mark.skip(reason="Known instability: <ISSUE-URL>")`

### Skipped Tests

**Allowed**: Conditional skips for missing optional dependencies

**Example**:
```python
@pytest.mark.validation
def test_dashboard_rendering():
    """Test dashboard visualization."""
    pytest.importorskip("matplotlib")
    # ... test code ...
```

**Not Allowed**: Skipping core functionality tests

---

## CI Integration

### PR Workflow (smoke only)
```yaml
# .github/workflows/ci-pr.yml
- name: Run smoke tests
  run: pytest -m "not validation" -v --tb=short
```

### Validation Workflow (separate)
```yaml
# .github/workflows/ci-validation.yml
- name: Run validation tests
  run: pytest -m validation -v
```

**Rationale**: Keep PR feedback fast (<1 min); run heavy validation on merge/schedule.

---

## Review Checklist

When reviewing test PRs, verify:

- [ ] Slow tests (>1s) are marked `@pytest.mark.validation`
- [ ] Tests are deterministic (explicit seed_all() calls)
- [ ] Smoke tests do not require optional dependencies
- [ ] New tests follow naming convention (`test_*.py`)
- [ ] Tests have docstrings explaining purpose
- [ ] Coverage does not decrease (unless intentional)
- [ ] No flaky tests (run multiple times to verify)

---

## References

- `pyproject.toml`: Pytest configuration
- `Makefile`: Test commands
- `docs/QUALITY_GATE_REPORT.md`: Baseline audit
- `tests/test_marker_enforcement.py`: CI guard test
- `tests/test_determinism.py`: Determinism validation
