"""Authentication routes."""

from __future__ import annotations

import hmac
import sqlite3
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, Form, Header, HTTPException, Request, Response, status

from bnsyn.web.config import Settings
from bnsyn.web.deps import get_current_user, get_db, get_settings
from bnsyn.web.models import TokenResponse, UserClaims
from bnsyn.web.security import create_access_token, derive_csrf_token, verify_password

router = APIRouter()


def _header_origin(value: str) -> str:
    parsed = urlparse(value)
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}".lower()


def _validate_logout_origin(request: Request, settings: Settings) -> None:
    if settings.environment not in {"dev", "prod"}:
        return
    allowed = {origin.lower() for origin in settings.allowed_origins}
    origin = request.headers.get("origin")
    referer = request.headers.get("referer")
    candidate = origin if origin is not None else referer
    if candidate is None:
        return
    if _header_origin(candidate) not in allowed:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Origin not allowed")


@router.post("/token", response_model=TokenResponse)
def issue_token(
    response: Response,
    username: str = Form(...),
    password: str = Form(...),
    settings: Settings = Depends(get_settings),
    conn: sqlite3.Connection = Depends(get_db),
) -> TokenResponse:
    """Issue JWT token for valid username/password credentials."""
    row = conn.execute(
        "SELECT id, email, password_hash, role, tenant_id FROM users WHERE email = ?", (username,)
    ).fetchone()
    if row is None or not verify_password(password, row["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    token, expires_in, jti, issued_at = create_access_token(
        settings=settings,
        conn=conn,
        user_id=str(row["id"]),
        email=str(row["email"]),
        role=str(row["role"]),
        tenant_id=str(row["tenant_id"]),
    )
    csrf_token = derive_csrf_token(
        secret=settings.jwt_secret,
        user_id=str(row["id"]),
        jti=jti,
        issued_at=issued_at,
    )
    response.set_cookie(
        key=settings.cookie_name,
        value=token,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        path="/",
        max_age=expires_in,
    )
    response.set_cookie(
        key=settings.csrf_cookie_name,
        value=csrf_token,
        httponly=False,
        secure=settings.cookie_secure,
        samesite="strict",
        path="/",
        max_age=expires_in,
    )
    return TokenResponse(access_token=token, token_type="bearer", expires_in=expires_in)


@router.post("/logout")
def logout(
    request: Request,
    response: Response,
    x_bnsyn_csrf: str | None = Header(default=None),
    user: UserClaims = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
    conn: sqlite3.Connection = Depends(get_db),
) -> dict[str, str]:
    """Revoke current token and clear auth cookie."""
    csrf_cookie = request.cookies.get(settings.csrf_cookie_name)
    if not csrf_cookie or not x_bnsyn_csrf or not hmac.compare_digest(csrf_cookie, x_bnsyn_csrf):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="CSRF validation failed")

    _validate_logout_origin(request, settings)

    conn.execute(
        "INSERT OR IGNORE INTO token_revocations (jti, revoked_at) VALUES (?, ?)",
        (user.jti, settings.timestamp_value),
    )
    conn.commit()
    response.delete_cookie(key=settings.cookie_name, path="/")
    response.delete_cookie(key=settings.csrf_cookie_name, path="/")
    return {"status": "ok"}
