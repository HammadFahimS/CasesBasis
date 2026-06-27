"""Apply pending SQL migrations to the database.

Usage:
    python scripts/migrate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.db import get_db_path, run_migrations


def main() -> None:
    print(f"Running migrations on: {get_db_path()}")
    run_migrations()
    print("Done.")


if __name__ == "__main__":
    main()
