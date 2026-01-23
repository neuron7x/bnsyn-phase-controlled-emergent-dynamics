# Normative vs Non‑Normative Documentation Policy

## Purpose
Prevent evidence drift: remove or downgrade claims that are not supported by primary sources.

## Required markup
- Put **[NORMATIVE]** at the start of any requirement.
- If the sentence contains a falsifiable quantitative claim, append `[CLM-XXXX]`.

## Examples
✅ Good:
- [NORMATIVE][CLM-0003] Use canonical NMDA Mg²⁺ block coefficients.

✅ Good (non-normative):
- [NON‑NORMATIVE] Some studies report large efficiency gains for sparse SNNs; values vary by hardware and workload.

❌ Bad:
- "SNNs are 6–8× more efficient" (no SSOT, no claim id, normative implied)
