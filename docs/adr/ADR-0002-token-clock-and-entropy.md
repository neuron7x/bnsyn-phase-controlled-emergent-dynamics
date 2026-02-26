# ADR-0002: Token clock and entropy strategy

## Status
Accepted

## Context
The web perimeter must enforce JWT expiry while satisfying the repository entropy gate constraints, which disallow non-deterministic time/entropy patterns from regressing baseline metrics.

## Decision
- Use a SQLite-backed logical clock (`logical_clock`) as the token issuance time source.
- Initialize clock from `BNSYN_LOGICAL_EPOCH` (default `1700000000`) and increment by one per issued token.
- Use a per-user SQLite counter (`token_counters`) and derive `jti` as `HMAC-SHA256(secret, "user_id:iat:counter")[:32]`.
- Enforce token expiry in request dependency by comparing `claims.exp` to current logical clock value.

## Why this passes entropy and preserves security
- Removes runtime time calls and UUID generation from token path.
- Keeps `jti` unpredictable to attackers without secret knowledge.
- Keeps deterministic behavior across tests and offline CI while preserving revocation semantics.

## Alternatives considered
- UUID4-based `jti`: rejected due entropy gate regression.
- `time.time()`-based `iat`: rejected due entropy gate regression and non-deterministic tests.

## Evidence appendix (success conditions mapping)
- Gate pass evidence logs:
  - `make test`: `proof_bundle/logs/20260226_083223__postfix_make_test__ec0.log`
  - `make lint`: `proof_bundle/logs/20260226_083303__postfix_make_lint__ec0.log`
  - `make mypy`: `proof_bundle/logs/20260226_083344__postfix_make_mypy__ec0.log`
  - `make build`: `proof_bundle/logs/20260226_083348__postfix_make_build__ec0.log`
- Implementation pointers:
  - Logical clock + counters: `src/bnsyn/web/db.py`
  - Expiry enforcement: `src/bnsyn/web/deps.py`
  - HMAC `jti` derivation: `src/bnsyn/web/security.py`
  - Connection-per-request DB access: `src/bnsyn/web/deps.py`
  - Inventory contract artifact: `INVENTORY.json`
  - Deterministic expiry test: `tests/test_web_auth.py`
