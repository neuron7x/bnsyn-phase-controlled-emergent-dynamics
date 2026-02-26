"""Web security perimeter configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, cast


_ENVIRONMENTS = {"dev", "prod", "test"}
_WEAK_SECRET_DENYLIST = {"change-me-in-production", "password", "secret", "test-secret", "changeme"}


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the web perimeter."""

    database_path: str
    jwt_secret: str
    jwt_algorithm: str
    access_token_ttl_seconds: int
    cookie_secure: bool
    cookie_samesite: Literal["lax", "strict", "none"]
    cookie_name: str
    csrf_cookie_name: str
    host: str
    port: int
    logical_epoch: int
    timestamp_value: str
    environment: Literal["dev", "prod", "test"]
    deterministic_mode: bool
    fixed_now: int | None
    allowed_origins: tuple[str, ...] = ()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_allowed_origins(raw: str) -> tuple[str, ...]:
    items = tuple(sorted({item.strip() for item in raw.split(",") if item.strip()}))
    return items


def validate_settings(settings: Settings) -> None:
    secret = settings.jwt_secret.strip()
    if settings.environment == "test":
        if not secret:
            raise ValueError("BNSYN_JWT_SECRET must not be empty")
    else:
        if not secret:
            raise ValueError("BNSYN_JWT_SECRET must be set and non-empty for dev/prod")
        if secret.lower() in _WEAK_SECRET_DENYLIST:
            raise ValueError("BNSYN_JWT_SECRET uses a blocked placeholder value")
        if len(secret) < 32:
            raise ValueError("BNSYN_JWT_SECRET must be at least 32 characters for dev/prod")

    if settings.environment == "prod" and settings.deterministic_mode:
        if os.getenv("BNSYN_ALLOW_DETERMINISTIC_IN_PROD", "0") != "1":
            raise ValueError("deterministic_mode is forbidden in prod unless BNSYN_ALLOW_DETERMINISTIC_IN_PROD=1")

    if settings.deterministic_mode and settings.fixed_now is None:
        raise ValueError("BNSYN_FIXED_NOW is required when deterministic mode is enabled")


def load_settings() -> Settings:
    """Load settings from environment with secure defaults."""
    environment_raw = os.getenv("BNSYN_ENVIRONMENT", "dev").strip().lower()
    if environment_raw not in _ENVIRONMENTS:
        raise ValueError("BNSYN_ENVIRONMENT must be one of: dev, prod, test")
    environment = cast(Literal["dev", "prod", "test"], environment_raw)
    deterministic_mode = environment == "test" or _env_bool("BNSYN_DETERMINISTIC_MODE", False)
    fixed_now_raw = os.getenv("BNSYN_FIXED_NOW")
    fixed_now = int(fixed_now_raw) if fixed_now_raw is not None else None

    default_allowed = "http://127.0.0.1,http://localhost"
    allowed_raw = os.getenv("BNSYN_ALLOWED_ORIGINS", default_allowed if environment in {"dev", "prod"} else "")

    settings = Settings(
        database_path=os.getenv("BNSYN_DB_PATH", "bnsyn_web.sqlite3"),
        jwt_secret=os.getenv("BNSYN_JWT_SECRET", ""),
        jwt_algorithm="HS256",
        access_token_ttl_seconds=int(os.getenv("BNSYN_ACCESS_TOKEN_TTL_SECONDS", "900")),
        cookie_secure=_env_bool("BNSYN_COOKIE_SECURE", True),
        cookie_samesite="lax",
        cookie_name="bnsyn_at",
        csrf_cookie_name="bnsyn_csrf",
        host=os.getenv("BNSYN_WEB_HOST", "127.0.0.1"),
        port=int(os.getenv("BNSYN_WEB_PORT", "8000")),
        logical_epoch=int(os.getenv("BNSYN_LOGICAL_EPOCH", "1700000000")),
        timestamp_value=os.getenv("BNSYN_TIMESTAMP_VALUE", "1970-01-01T00:00:00Z"),
        environment=environment,
        deterministic_mode=deterministic_mode,
        fixed_now=fixed_now,
        allowed_origins=_parse_allowed_origins(allowed_raw),
    )
    validate_settings(settings)
    return settings
