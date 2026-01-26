# BN-Syn Temperature Ablation Hypothesis

**Status**: NON-NORMATIVE experimental document  
**Version**: 1.0  
**Date**: 2026-01-26

This document describes a falsifiable hypothesis linking temperature-controlled plasticity gating to consolidation stability in dual-weight synapses.

---

## Hypothesis H1: Temperature-Controlled Consolidation Stability

**Statement**: Phase-controlled temperature gating improves consolidation stability in DualWeights compared to fixed temperature regimes.

**Rationale**: The temperature schedule (`TemperatureSchedule`) modulates plasticity via `gate_sigmoid(T, Tc, gate_tau)`, which gates the effective learning rate applied to `DualWeights.w_fast`. Geometric cooling (`T0 → Tmin` with `alpha < 1`) is expected to provide a controlled annealing process that reduces variance in final consolidated weights (`w_cons`) across independent trials compared to fixed high or low temperature regimes.

---

## Experimental Design

### Conditions

Four temperature regimes are tested under identical synthetic input patterns:

| Condition | Description | Temperature Profile |
|-----------|-------------|---------------------|
| **cooling_geometric** | Geometric cooling | T(step) = max(Tmin, T(step-1) * alpha) starting from T0 |
| **fixed_high** | Fixed high temperature | T(step) = T0 for all steps |
| **fixed_low** | Fixed low temperature | T(step) = Tmin for all steps |
| **random_T** | Random temperature | T(step) ~ Uniform(Tmin, T0), seeded |

### Synthetic Input Protocol

- **Input pattern**: Deterministic pseudo-random pulses to `fast_update` using seeded `numpy.random.Generator`.
- **Steps**: 5000 consolidation steps (default configuration).
- **Seeds**: 20 independent trials per condition (validation run); 5 for smoke tests.
- **DualWeightParams**: Default parameters from `bnsyn.config.DualWeightParams`.
- **TemperatureParams**: T0=1.0, Tmin=1e-3, alpha=0.95, Tc=0.1, gate_tau=0.02.

### Effective Update Rule

At each step, the temperature gate modulates the fast weight update:

```
gate = gate_sigmoid(T, Tc, gate_tau)
effective_update = gate * fast_update
DualWeights.step(dt_s, params, effective_update)
```

---

## Metrics

Stability metrics are computed across the `seeds` trials for each condition:

| Metric ID | Definition | Acceptance Target |
|-----------|------------|-------------------|
| **stability_w_total_var_end** (PRIMARY) | Variance across seeds of final `mean(w_total)` | Lower for cooling vs fixed_high |
| **stability_w_cons_var_end** (SECONDARY) | Variance across seeds of final `mean(w_cons)` | Lower for cooling vs fixed_high |
| **tag_activity_mean** | Mean fraction of active tags over time | Reported (no specific target) |
| **protein_mean_end** | Final protein level (mean across seeds) | Reported (no specific target) |

**Note on w_cons metric**: Consolidation requires both synaptic tags AND protein synthesis (cooperative threshold N_p=50 simultaneous tags). In conditions with low plasticity (cooling, fixed_low), tag counts may remain below threshold, preventing protein synthesis and thus keeping w_cons at baseline. Therefore, **w_total variance is the primary stability metric**, as it captures total synaptic weight stability regardless of consolidation state.

### Acceptance Criterion

**H1 is supported** if:
- `cooling_geometric` produces **lower** `stability_w_total_var_end` than `fixed_high` by at least 10% (relative reduction).

**H1 is strongly supported** if:
- `cooling_geometric` produces **lower** `stability_w_cons_var_end` than `fixed_high` (when consolidation occurs in both conditions).

**H1 is refuted** if the above conditions do not hold.

---

## Reproduce

### Installation

```bash
pip install -e ".[dev,viz]"
```

### Run flagship experiment

```bash
# Full validation run (seeds=20, ~2-5 minutes)
python -m experiments.runner temp_ablation_v1

# Fast smoke test (seeds=5)
python -m experiments.runner temp_ablation_v1 --seeds 5 --out results/_smoke
```

### Generate visualizations

```bash
python scripts/visualize_experiment.py --run-id temp_ablation_v1
```

### Verify hypothesis

```bash
python -m experiments.verify_hypothesis docs/HYPOTHESIS.md results/temp_ablation_v1
```

---

## Expected Outputs

- **results/temp_ablation_v1/**: Per-condition JSON files with per-seed metrics + aggregates.
- **results/temp_ablation_v1/manifest.json**: Reproducibility manifest (git commit, params, hashes).
- **figures/hero.png**: Stability curve comparison across conditions.
- **figures/temperature_vs_stability.png**: Temperature profile vs stability metrics.
- **figures/tag_activity.png**: Tag activity over time by condition.
- **figures/comparison_grid.png**: Multi-panel comparison grid.

---

## References

- SPEC P1-5: Temperature schedule and gating (`src/bnsyn/temperature/schedule.py`).
- SPEC P1-6: Dual-weight consolidation (`src/bnsyn/consolidation/dual_weight.py`).
- SPEC P2-9: Determinism protocol (`src/bnsyn/rng.py`).

---

## Notes

This document is **non-governed** and does not use normative keywords (`must`, `shall`, `required`, `guarantee`) except in clearly marked quotations or references to governed documents. It describes an experimental protocol for generating evidence artifacts.
