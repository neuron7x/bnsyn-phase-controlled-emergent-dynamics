"""SQLite helpers for web perimeter."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator


SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS tenants (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL,
        tenant_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (tenant_id) REFERENCES tenants(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS token_revocations (
        jti TEXT PRIMARY KEY,
        revoked_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS logical_clock (
        id INTEGER PRIMARY KEY CHECK(id = 1),
        now INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS token_counters (
        user_id TEXT PRIMARY KEY,
        counter INTEGER NOT NULL
    )
    """,
)


DEFAULT_TIMESTAMP = "1970-01-01T00:00:00Z"


def connect_db(path: str) -> sqlite3.Connection:
    """Open SQLite connection with row mapping."""
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def bootstrap_schema(conn: sqlite3.Connection) -> None:
    """Create schema idempotently."""
    for statement in SCHEMA_STATEMENTS:
        conn.execute(statement)
    conn.commit()


def ensure_logical_clock(conn: sqlite3.Connection, *, epoch: int) -> int:
    """Return current logical clock value, initializing if missing."""
    row = conn.execute("SELECT now FROM logical_clock WHERE id = 1").fetchone()
    if row is None:
        conn.execute("INSERT INTO logical_clock (id, now) VALUES (1, ?)", (epoch,))
        conn.commit()
        return epoch
    return int(row["now"])


def next_logical_time(conn: sqlite3.Connection, *, epoch: int) -> int:
    """Advance and return the logical clock time."""
    current = ensure_logical_clock(conn, epoch=epoch)
    next_value = current + 1
    conn.execute("UPDATE logical_clock SET now = ? WHERE id = 1", (next_value,))
    conn.commit()
    return next_value


def next_token_counter(conn: sqlite3.Connection, *, user_id: str) -> int:
    """Advance and return per-user token counter."""
    row = conn.execute("SELECT counter FROM token_counters WHERE user_id = ?", (user_id,)).fetchone()
    if row is None:
        conn.execute("INSERT INTO token_counters (user_id, counter) VALUES (?, 1)", (user_id,))
        conn.commit()
        return 1
    counter = int(row["counter"]) + 1
    conn.execute("UPDATE token_counters SET counter = ? WHERE user_id = ?", (counter, user_id))
    conn.commit()
    return counter


def insert_tenant(conn: sqlite3.Connection, *, tenant_id: str, name: str, created_at: str = DEFAULT_TIMESTAMP) -> None:
    """Insert a tenant record."""
    conn.execute(
        "INSERT INTO tenants (id, name, created_at) VALUES (?, ?, ?)",
        (tenant_id, name, created_at),
    )


def insert_user(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    email: str,
    password_hash: str,
    role: str,
    tenant_id: str,
    created_at: str = DEFAULT_TIMESTAMP,
) -> None:
    """Insert a user record."""
    conn.execute(
        "INSERT INTO users (id, email, password_hash, role, tenant_id, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, email, password_hash, role, tenant_id, created_at),
    )


@contextmanager
def db_connection(path: str) -> Iterator[sqlite3.Connection]:
    """Context manager for SQLite connection."""
    conn = connect_db(path)
    try:
        yield conn
    finally:
        conn.close()
