# Architecture (repo-aligned)

This page is a structural map of runnable surfaces and governance gates in this repository.
For public surfaces and SSOT enforcement details, see [PROJECT_SURFACES.md](PROJECT_SURFACES.md) and [ENFORCEMENT_MATRIX.md](ENFORCEMENT_MATRIX.md).

## Diagram A — System flow (CLI → runtime → experiments → artifacts → validators/CI)

```mermaid
flowchart LR
  CLI[src/bnsyn/cli.py] --> RUNTIME[src/bnsyn/**]
  CLI --> EXP[experiments/**]
  RUNTIME --> ART1[results/**]
  RUNTIME --> ART2[figures/**]
  EXP --> ART1
  EXP --> ART2
  DOCS[docs/**]
  CLAIMS[claims/claims.yml]
  BIB[bibliography/**]
  SCHEMAS[schemas/*.json]
  DOCS --> VAL[scripts/validate_traceability.py\nscripts/check_internal_links.py\nscripts/discover_public_surfaces.py]
  CLAIMS --> VAL
  BIB --> VAL
  SCHEMAS --> VAL
  VAL --> CI[.github/workflows/ci-pr.yml\n.github/workflows/ci-validation.yml\n.github/workflows/ci-pr-atomic.yml]
  CI --> VAL
```

## Diagram B — Governance / SSOT enforcement flow

```mermaid
flowchart TD
  SSOT1[docs/TRACEABILITY.md]
  SSOT2[docs/ENFORCEMENT_MATRIX.md]
  SSOT3[docs/PROJECT_SURFACES.md]
  SSOT4[claims/claims.yml + bibliography/**]
  RULES[scripts/ssot_rules.py]
  V1[scripts/validate_traceability.py]
  V2[scripts/check_internal_links.py]
  V3[scripts/discover_public_surfaces.py --check]
  G1[.github/workflows/ci-pr.yml]
  G2[.github/workflows/ci-validation.yml]
  G3[.github/workflows/ci-pr-atomic.yml]

  SSOT1 --> V1
  SSOT2 --> RULES
  SSOT3 --> V3
  SSOT4 --> V1
  RULES --> G1
  V1 --> G2
  V2 --> G2
  V3 --> G3
```

## How to read these diagrams

- Nodes are concrete repo paths, never conceptual-only components.
- Solid arrows mean dependency or enforcement direction.
- `src/bnsyn/cli.py` is the documented command entry surface.
- `src/bnsyn/**` and `experiments/**` feed generated artifacts in `results/**` and `figures/**`.
- SSOT inputs are documentation, schemas, and claim/evidence files.
- Validators under `scripts/**` are the executable enforcement layer.
- CI workflows under `.github/workflows/**` are gate runners, not alternate policy definitions.
- `docs/INDEX.md` remains the canonical navigation hub for all docs.
