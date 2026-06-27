from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

_local = threading.local()


def get_db_path() -> str:
    return os.getenv("DB_PATH", "casesbasis.db")


def get_db() -> sqlite3.Connection:
    """Return a per-thread SQLite connection, creating it on first use."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        # isolation_level=None puts sqlite3 in autocommit mode. Combined with
        # explicit BEGIN IMMEDIATE in write_transaction(), this prevents
        # SQLITE_BUSY_SNAPSHOT — a WAL-mode error that bypasses busy_timeout
        # entirely and fires immediately when a connection holds a stale read
        # snapshot while another writer has advanced the WAL.
        conn = sqlite3.connect(get_db_path(), check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA synchronous=NORMAL")
        _local.conn = conn
    return conn


@contextmanager
def write_transaction() -> Generator[sqlite3.Connection, None, None]:
    """Wrap writes in BEGIN IMMEDIATE so busy_timeout retries work properly.

    In WAL mode, BEGIN DEFERRED (Python's default) can escalate to a write lock
    while holding a read snapshot, triggering SQLITE_BUSY_SNAPSHOT which bypasses
    the busy handler completely. BEGIN IMMEDIATE acquires the write lock upfront,
    so all contention goes through the busy_timeout retry path.
    """
    conn = get_db()
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
        conn.execute("COMMIT")
    except BaseException:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_migrations() -> None:
    migrations_dir = Path(__file__).parent.parent / "migrations"
    conn = get_db()
    # DDL and INSERT are autocommit in isolation_level=None mode.
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations "
        "(filename TEXT PRIMARY KEY, applied_at TEXT NOT NULL)"
    )

    applied = {row[0] for row in conn.execute("SELECT filename FROM schema_migrations")}
    for sql_file in sorted(migrations_dir.glob("*.sql")):
        if sql_file.name in applied:
            continue
        # executescript() commits any open transaction, then runs the script.
        conn.executescript(sql_file.read_text())
        conn.execute(
            "INSERT INTO schema_migrations (filename, applied_at) VALUES (?, ?)",
            (sql_file.name, utcnow()),
        )
