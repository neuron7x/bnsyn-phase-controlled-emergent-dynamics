## Description

Provide a clear and concise description of your changes.

## Type of Change

- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update
- [ ] 🏗️ Infrastructure/CI change
- [ ] 🧪 Test improvement

## Pre-Merge Checklist

**REQUIRED before creating PR:**

### Local Verification
- [ ] Ran `pre-commit run --all-files` ✅
- [ ] Ran `make check` (ruff, mypy, pylint) ✅
- [ ] Ran `pytest -m "not validation" --cov=src/bnsyn --cov-fail-under=85` ✅
- [ ] Code coverage ≥85% ✅
- [ ] No linter errors ✅
- [ ] mypy --strict passed ✅

### SSOT Gates (Single Source of Truth)
- [ ] Validated bibliography: `python scripts/validate_bibliography.py` ✅
- [ ] Validated claims: `python scripts/validate_claims.py` ✅
- [ ] Scanned governed docs: `python scripts/scan_governed_docs.py` ✅
- [ ] Scanned normative tags: `python scripts/scan_normative_tags.py` ✅

### Determinism (A1: 96%)
- [ ] Used `seed_all()` for any random operations ✅
- [ ] Verified determinism (3x runs with same seed produce identical outputs) ✅
- [ ] No global numpy RNG usage ✅

### Documentation (A7: 90%)
- [ ] All new functions have docstrings (Google style) ✅
- [ ] Updated `docs/SPEC.md` if changing specifications ✅
- [ ] Updated `README.md` if changing user-facing features ✅

### Security (A6: 90%)
- [ ] No secrets committed (verified with gitleaks) ✅
- [ ] Ran `pip-audit` (no vulnerabilities) ✅
- [ ] Ran `bandit -r src/ -ll` (no high/medium issues) ✅

## Testing

**Categories tested:**
- [ ] Unit tests
- [ ] Integration tests
- [ ] Property-based tests (Hypothesis)
- [ ] Validation tests (large N, statistical)
- [ ] Benchmarks (performance)

**Commands run:**
```bash
# Example:
pytest tests/test_neuron.py -v
pytest tests/test_determinism.py -v --count 3  # 3x for determinism
```

## Performance Impact

- [ ] No performance impact
- [ ] Performance improved (provide benchmarks)
- [ ] Performance degraded (justify and provide mitigation)

**Benchmarks (if applicable):**
```
# Before: X ms
# After: Y ms
```

## Breaking Changes

- [ ] No breaking changes
- [ ] Breaking changes (list below):

**Migration guide (if breaking):**
```
# How to update existing code
```

## Reproducibility Commands

Provide exact commands to reproduce your changes:

```bash
# Clone and setup
git clone https://github.com/neuron7x/bnsyn-phase-controlled-emergent-dynamics.git
cd bnsyn-phase-controlled-emergent-dynamics
git checkout <YOUR_BRANCH>

# Install dependencies
pip install -e ".[dev,test,viz]"

# Run specific tests
pytest tests/... -v

# Verify determinism
pytest tests/test_determinism.py -v --count 3
```

## Checklist for Reviewer

**Axiom Compliance:**
- [ ] A1 (Determinism): seed_all() used, no global RNG
- [ ] A2 (Composability): Reusable functions/classes, no tight coupling
- [ ] A3 (Observability): Logging, error messages, docstrings
- [ ] A4 (Exhaustiveness): Edge cases tested, coverage ≥85%
- [ ] A5 (Performance): No unnecessary loops, efficient algorithms
- [ ] A6 (Security): No secrets, no unsafe operations
- [ ] A7 (Documentation): Docstrings, README updates, SPEC.md updates

**Code Quality:**
- [ ] Code is readable and maintainable
- [ ] No unnecessary complexity
- [ ] Follows existing code style
- [ ] Tests are clear and comprehensive

## Related Issues

Closes #(issue number)

## Additional Notes

Any additional information that would be helpful for reviewers.
