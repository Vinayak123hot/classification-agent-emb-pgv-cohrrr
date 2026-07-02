"""
db.py — Postgres (pgvector) access, as a class.

`Database` owns ONE psycopg2 connection for the process (the service runs a
single worker) and reconnects transparently if it drops. It takes a Config so the
connection string is resolved from Key Vault / env at first use.
"""
from __future__ import annotations

import logging
import threading

import psycopg2

logger = logging.getLogger("kb-tool.db")


class Database:
    def __init__(self, config):
        self._config = config                 # provides get_pg_conn()
        self._conn = None                      # lazily opened psycopg2 connection
        self._lock = threading.Lock()          # guards (re)connection

    def _connect(self):
        """Open a fresh autocommit connection using the resolved DSN."""
        dsn = self._config.get_pg_conn()
        if not dsn:
            raise RuntimeError(
                "No Postgres connection string. Set PG_CONN or the Key Vault "
                "secret 'pg-vector-conn'."
            )
        conn = psycopg2.connect(dsn, connect_timeout=30)
        conn.autocommit = True                 # each statement commits immediately
        return conn

    def get_conn(self):
        """Return a live connection, (re)connecting if needed."""
        with self._lock:
            if self._conn is None or self._conn.closed:
                self._conn = self._connect()
                return self._conn
        # cheap liveness probe outside the lock
        try:
            with self._conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()
        except Exception:
            logger.warning("Postgres connection lost — reconnecting.")
            with self._lock:
                self._conn = self._connect()
        return self._conn

    def fetch_all(self, sql: str, params: tuple = ()) -> list[tuple]:
        """Run a SELECT and return all rows."""
        with self.get_conn().cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def execute(self, sql: str, params: tuple = ()) -> None:
        """Run a statement with no result set."""
        with self.get_conn().cursor() as cur:
            cur.execute(sql, params)
