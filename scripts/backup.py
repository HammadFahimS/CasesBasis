"""Database backup script.

Usage:
    python scripts/backup.py

Creates a timestamped copy of casesbasis.db in the backups/ directory.
Safe to run while the app is running — SQLite WAL mode allows consistent
hot backups via the backup API.

Cron example (every 6 hours):
    0 */6 * * * cd /path/to/CasesBasis && .venv/bin/python scripts/backup.py
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
BACKUP_DIR = ROOT / "backups"


def backup() -> Path:
    from backend.db import get_db_path

    src = Path(get_db_path())
    if not src.exists():
        print(f"Database not found at {src}", file=sys.stderr)
        sys.exit(1)

    BACKUP_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = BACKUP_DIR / f"casesbasis_{ts}.db"

    # SQLite online backup API — consistent snapshot even under concurrent writes
    src_conn = sqlite3.connect(str(src))
    dest_conn = sqlite3.connect(str(dest))
    try:
        src_conn.backup(dest_conn)
    finally:
        dest_conn.close()
        src_conn.close()

    size_kb = dest.stat().st_size // 1024
    print(f"Backup written to {dest} ({size_kb} KB)")
    return dest


if __name__ == "__main__":
    backup()
