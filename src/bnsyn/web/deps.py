"""FastAPI dependencies for authentication, RBAC, and tenant context."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast

from fastapi import Depends, HTTPException, Request, status

from bnsyn.web.config import Settings
from bnsyn.web.db import connect_db
from bnsyn.web.models import UserClaims
from bnsyn.web.security import decode_access_token, role_allows


def get_settings(request: Request) -> Settings:
    """Resolve app settings from request state."""
    return cast(Settings, request.app.state.settings)


@contextmanager
def _db_conn(path: str) -> Iterator[sqlite3.Connection]:
    conn = connect_db(path)
    try:
        yield conn
    finally:
        conn.close()


def get_db(settings: Settings = Depends(get_settings)) -> Iterator[sqlite3.Connection]:
    """Resolve app database connection from request state."""
    with _db_conn(settings.database_path) as conn:
        yield conn


def get_current_user(
    request: Request,
    settings: Settings = Depends(get_settings),
    conn: sqlite3.Connection = Depends(get_db),
) -> UserClaims:
    """Resolve current user from JWT cookie and revocation table."""
    token = request.cookies.get(settings.cookie_name)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    try:
        claims = decode_access_token(token, settings)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc


    revoked = conn.execute("SELECT 1 FROM token_revocations WHERE jti = ?", (claims.jti,)).fetchone()
    if revoked:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token revoked")

    return claims


def require_role(min_role: str) -> Callable[[UserClaims], UserClaims]:
    """Build RBAC dependency with minimum role."""

    def _dep(user: UserClaims = Depends(get_current_user)) -> UserClaims:
        if not role_allows(actual_role=user.role, min_role=min_role):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
        return user

    return _dep


def get_tenant_id(user: UserClaims = Depends(get_current_user)) -> str:
    """Resolve tenant context from authenticated identity."""
    return user.tenant_id
