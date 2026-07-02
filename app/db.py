"""
db.py — thin psycopg2 connection helper for the kbtool Postgres (pgvector) DB.

The connection string (a libpq URI with sslmode=require) comes from Key Vault
secret `pg-vector-conn`, or the PG_CONN env var for local runs. We keep ONE
module-level connection for the FastAPI process (single worker) and reconnect
transparently if it has dropped.
"""
from __future__ import annotations

import logging
import threading

import psycopg2

from config import get_pg_conn

logger = logging.getLogger("kb-tool.db")

_conn = None
_lock = threading.Lock()


def _connect():
    dsn = get_pg_conn()
    if not dsn:
        raise RuntimeError(
            "No Postgres connection string. Set PG_CONN or the Key Vault secret "
            "'pg-vector-conn'."
        )
    c = psycopg2.connect(dsn, connect_timeout=30)
    c.autocommit = True
    return c


def get_conn():
    """Return a live, autocommit connection, (re)connecting if needed."""
    global _conn
    with _lock:
        if _conn is None or _conn.closed:
            _conn = _connect()
            return _conn
    # cheap liveness check outside the lock
    try:
        with _conn.cursor() as cur:
            cur.execute("SELECT 1;")
            cur.fetchone()
    except Exception:
        logger.warning("Postgres connection lost — reconnecting.")
        with _lock:
            _conn = _connect()
    return _conn


def fetch_all(sql: str, params: tuple = ()) -> list[tuple]:
    with get_conn().cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def execute(sql: str, params: tuple = ()) -> None:
    with get_conn().cursor() as cur:
        cur.execute(sql, params)
