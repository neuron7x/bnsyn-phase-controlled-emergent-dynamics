"""Admin routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from bnsyn.web.deps import get_tenant_id, require_role
from bnsyn.web.models import UserClaims

router = APIRouter(prefix="/admin")


@router.get("/ping")
def admin_ping(
    user: UserClaims = Depends(require_role("admin")),
    tenant_id: str = Depends(get_tenant_id),
) -> dict[str, str]:
    """Admin-only ping endpoint."""
    return {"status": "ok", "user_id": user.sub, "tenant_id": tenant_id}
