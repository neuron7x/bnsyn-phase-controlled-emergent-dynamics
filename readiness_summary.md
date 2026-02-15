Readiness: 43% (Dev-Ready (unstable)); confidence: 1.00.
Top blockers:
1) Scope mismatch: repo is simulation-focused, not marketplace product.
2) Local build failed (`python -m build` missing module).
3) Local tests failed at collection (missing hypothesis/pyyaml/psutil deps).
4) No auth/RBAC/multi-tenant controls for SaaS flows.
5) No payment→entitlement→download pipeline or idempotent transaction model.
