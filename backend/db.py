from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def get_db_path() -> str:
    return os.getenv("DB_PATH", "casesbasis.db")


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_migrations() -> None:
    migrations_dir = Path(__file__).parent.parent / "migrations"
    conn = get_db()
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(filename TEXT PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        conn.commit()

        applied = {row[0] for row in conn.execute("SELECT filename FROM schema_migrations")}
        for sql_file in sorted(migrations_dir.glob("*.sql")):
            if sql_file.name in applied:
                continue
            conn.executescript(sql_file.read_text())
            conn.execute(
                "INSERT INTO schema_migrations (filename, applied_at) VALUES (?, ?)",
                (sql_file.name, utcnow()),
            )
            conn.commit()
    finally:
        conn.close()
