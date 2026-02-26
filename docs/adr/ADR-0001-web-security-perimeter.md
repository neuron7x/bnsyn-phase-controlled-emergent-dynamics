# ADR-0001: Web Security Perimeter for `bnsyn`

## Status
Accepted

## Context
`bnsyn` requires a production-grade web security perimeter covering authN, authZ, session invalidation, RBAC, tenant context, and health/readiness endpoints without external services.

## Decision
- Use FastAPI with OAuth2 password credentials endpoint (`POST /token`) returning HS256 JWT access token.
- Use `PyJWT` for HS256 token signing/verification; reject `python-jose` to keep dependency surface minimal.
- Transport JWT in HttpOnly cookie (`bnsyn_at`, SameSite=Lax, Secure configurable).
- Use JWT + revocation table (`token_revocations` keyed by `jti`) rather than server sessions:
  - preserves stateless verification path,
  - supports explicit logout invalidation,
  - keeps service horizontally simple without sticky session state.
- Use stdlib `sqlite3` for persistence rather than SQLAlchemy:
  - minimizes dependency surface,
  - satisfies local/offline execution constraints,
  - sufficient for current perimeter scope.
- Role ladder is strict and ordered: `owner > admin > member > viewer`.
- Access token lifetime is 15 minutes (`900s`), with revocation performed by storing `jti` and denying any revoked token on subsequent requests.

## Consequences
- Token verification is lightweight and local; logout is durable via revocation table.
- SQLite is simple and deterministic but not intended for distributed write-heavy deployments.
- RBAC and tenant context are uniformly enforced with dependency injection on protected routes.

## Alternatives Rejected
- Server-side session store only: rejected to avoid additional state management complexity.
- SQLAlchemy ORM: rejected in favor of stdlib `sqlite3` to keep runtime dependencies minimal.
- `python-jose`: rejected because `PyJWT` sufficiently covers HS256 requirements with fewer moving parts.

## Security Implications
- Login errors are generic (`401 Invalid credentials`) to prevent user enumeration.
- HttpOnly cookie transport reduces token exfiltration via JavaScript.
- SameSite=Lax and configurable Secure flag support browser CSRF/cookie posture hardening.
- Revocation table (`token_revocations`) provides server-enforced logout invalidation via `jti`.
