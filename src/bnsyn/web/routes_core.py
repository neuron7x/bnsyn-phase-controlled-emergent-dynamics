"""Core health/readiness/profile routes."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends

from bnsyn.web.deps import get_current_user, get_db, get_tenant_id
from bnsyn.web.models import MeResponse, UserClaims

router = APIRouter()


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness endpoint."""
    return {"status": "ok"}


@router.get("/readyz")
def readyz(conn: sqlite3.Connection = Depends(get_db)) -> dict[str, str]:
    """Readiness endpoint verifying DB reachability."""
    conn.execute("SELECT 1").fetchone()
    return {"status": "ready"}


@router.get("/me", response_model=MeResponse)
def me(user: UserClaims = Depends(get_current_user), tenant_id: str = Depends(get_tenant_id)) -> MeResponse:
    """Authenticated user profile endpoint."""
    return MeResponse(user_id=user.sub, email=user.email, role=user.role, tenant_id=tenant_id)
