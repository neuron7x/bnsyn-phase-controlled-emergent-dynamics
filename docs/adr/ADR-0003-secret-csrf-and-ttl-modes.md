# ADR-0003: Secret Policy, CSRF Double-Submit, and TTL Modes

## Status
Accepted

## Context
The web perimeter uses cookie-based JWT auth. Three blockers were identified:
1. insecure default JWT secret,
2. missing CSRF protection for cookie-authenticated logout,
3. TTL semantics that must be real-time in dev/prod while preserving deterministic CI behavior.

## Decision
- **Secret fail-fast policy**
  - `BNSYN_JWT_SECRET` is required for `dev` and `prod`.
  - Reject empty/whitespace, length `< 32`, and denylist placeholders.
  - `test` environment allows `test-secret` for deterministic local/CI tests.
  - CLI returns exit code `2` on settings validation error.

- **CSRF posture for `/logout`**
  - `/token` sets:
    - `bnsyn_at` (HttpOnly) auth cookie,
    - `bnsyn_csrf` (non-HttpOnly, `SameSite=Strict`) CSRF cookie.
  - CSRF token is deterministic: `HMAC_SHA256(jwt_secret, "{user_id}:{jti}:{iat}")[:32]`.
  - `/logout` requires header `x-bnsyn-csrf` matching cookie `bnsyn_csrf`; mismatch returns `403`.

- **TTL modes**
  - `dev`/`prod` default to real time: `iat=int(time.time())`, `exp=iat+ttl`, JWT decode enforces `exp` normally.
  - Deterministic mode for test/offline (`environment=test` or `BNSYN_DETERMINISTIC_MODE=1`) uses `fixed_now` for both issuance and verification.
  - JTI remains entropy-safe and deterministic using HMAC with per-user `token_counters` in SQLite.

## Consequences
- Production defaults become fail-closed and time-correct.
- Logout CSRF surface is materially reduced without adding external services.
- CI/test runs retain deterministic behavior required by entropy/reproducibility gates.
