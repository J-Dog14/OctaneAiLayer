"""
Database connection helpers.

- backend_conn(): writable connection to your backend Neon DB
                 (analytics, public, ai_layer, app_db_snapshot schemas live here)
- app_db_conn():  read-only connection to the App DB backup branch

Both are context managers that handle commit/rollback/close for you.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterable

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required env var: {name}. Check your .env file.")
    return val


@contextmanager
def backend_conn():
    """Read/write connection to the backend Neon DB."""
    conn = psycopg2.connect(_require_env("BACKEND_DB_URL"))
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def app_db_conn():
    """Read-only connection to the App DB backup branch.

    `set_session(readonly=True)` makes the SERVER reject any write attempt — a
    belt-and-suspenders guard on top of pointing at the backup branch.
    """
    conn = psycopg2.connect(_require_env("APP_DB_URL"))
    conn.set_session(readonly=True)
    try:
        yield conn
    finally:
        conn.close()


def query(conn, sql: str, params: Iterable[Any] | None = None) -> list[dict]:
    """Run a SELECT and return rows as a list of dicts."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params or [])
        return cur.fetchall()


def exec_sql(conn, sql: str, params: Iterable[Any] | None = None) -> None:
    """Run a statement that doesn't return rows (INSERT/UPDATE/DDL)."""
    with conn.cursor() as cur:
        cur.execute(sql, params or [])


def returning_id(conn, sql: str, params: Iterable[Any] | None = None) -> int:
    """Run an INSERT ... RETURNING id and return that id."""
    with conn.cursor() as cur:
        cur.execute(sql, params or [])
        return cur.fetchone()[0]
