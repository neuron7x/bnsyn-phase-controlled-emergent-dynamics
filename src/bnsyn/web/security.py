"""Security primitives for password and JWT handling."""

from __future__ import annotations

import hmac
import importlib
import sqlite3
from datetime import UTC, datetime
from hashlib import sha256
from typing import cast

import jwt

from bnsyn.web.config import Settings
from bnsyn.web.db import next_token_counter
from bnsyn.web.models import UserClaims

CryptContext = getattr(importlib.import_module("passlib.context"), "CryptContext")
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
ROLE_ORDER: dict[str, int] = {"viewer": 1, "member": 2, "admin": 3, "owner": 4}


def hash_password(password: str) -> str:
    """Hash a plaintext password using argon2id."""
    return cast(str, pwd_context.hash(password))


def verify_password(password: str, password_hash: str) -> bool:
    """Validate a plaintext password against stored hash."""
    return cast(bool, pwd_context.verify(password, password_hash))


def _derive_jti(*, secret: str, user_id: str, issued_at: int, counter: int) -> str:
    material = f"{user_id}:{issued_at}:{counter}".encode("utf-8")
    return hmac.new(secret.encode("utf-8"), material, sha256).hexdigest()[:32]


def derive_csrf_token(*, secret: str, user_id: str, jti: str, issued_at: int) -> str:
    """Derive deterministic CSRF token bound to user/token/time."""
    material = f"{user_id}:{jti}:{issued_at}".encode("utf-8")
    return hmac.new(secret.encode("utf-8"), material, sha256).hexdigest()[:32]


def _issued_at(settings: Settings) -> int:
    if settings.deterministic_mode:
        if settings.fixed_now is None:
            raise ValueError("fixed_now is required in deterministic mode")
        return settings.fixed_now
    return int(datetime.now(tz=UTC).timestamp())


def create_access_token(
    *,
    settings: Settings,
    conn: sqlite3.Connection,
    user_id: str,
    email: str,
    role: str,
    tenant_id: str,
) -> tuple[str, int, str, int]:
    """Create a signed JWT access token."""
    issued_at = _issued_at(settings)
    counter = next_token_counter(conn, user_id=user_id)
    jti = _derive_jti(secret=settings.jwt_secret, user_id=user_id, issued_at=issued_at, counter=counter)
    expires_at = issued_at + settings.access_token_ttl_seconds
    claims = {
        "sub": user_id,
        "email": email,
        "role": role,
        "tenant_id": tenant_id,
        "jti": jti,
        "iat": issued_at,
        "exp": expires_at,
    }
    token = jwt.encode(claims, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return token, settings.access_token_ttl_seconds, jti, issued_at


def decode_access_token(token: str, settings: Settings) -> UserClaims:
    """Decode and validate a signed JWT access token."""
    if settings.deterministic_mode:
        decoded = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            options={"verify_exp": False, "verify_iat": False},
        )
        claims = UserClaims.model_validate(decoded)
        if settings.fixed_now is None or claims.exp <= settings.fixed_now:
            raise jwt.InvalidTokenError("Token expired")
        return claims
    decoded = jwt.decode(
        token,
        settings.jwt_secret,
        algorithms=[settings.jwt_algorithm],
        options={"verify_iat": False},
    )
    return UserClaims.model_validate(decoded)


def role_allows(*, actual_role: str, min_role: str) -> bool:
    """Compare role rank for RBAC checks."""
    return ROLE_ORDER.get(actual_role, 0) >= ROLE_ORDER.get(min_role, 0)
