"""Pydantic models for web perimeter."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base strict model."""

    model_config = ConfigDict(strict=True, extra="forbid")


class TokenResponse(StrictModel):
    """OAuth2 token response."""

    access_token: str
    token_type: str
    expires_in: int


class MeResponse(StrictModel):
    """Identity response."""

    user_id: str
    email: str
    role: str
    tenant_id: str


class UserClaims(StrictModel):
    """Validated token claims."""

    sub: str
    email: str
    role: str
    tenant_id: str
    jti: str
    iat: int
    exp: int
