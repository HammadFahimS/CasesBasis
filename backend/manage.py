"""Management CLI for CasesBasis backend operations.

Usage:
    python -m backend.manage migrate-cases   # backfill case_json schema_version
    python -m backend.manage migrate-cases --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv


def cmd_migrate_cases(dry_run: bool = False) -> int:
    """Apply case_migrations.migrate_case_json to every row in simulated_cases.

    Reads every case, runs the migration chain, and writes back rows where the
    dict changed (idempotent — rows already at CASE_SCHEMA_VERSION are skipped).
    Use ``--dry-run`` to preview which rows would be updated."""
    from backend.case_migrations import CASE_SCHEMA_VERSION, migrate_case_json
    from backend.db import get_db, utcnow, write_transaction

    conn = get_db()
    rows = conn.execute("SELECT id, case_json FROM simulated_cases").fetchall()
    updated = 0
    for row in rows:
        raw = json.loads(row["case_json"])
        migrated = migrate_case_json(raw)
        if migrated == raw:
            continue
        updated += 1
        print(f"  {'[dry-run] ' if dry_run else ''}migrate {row['id']} "
              f"(schema_version {raw.get('schema_version', 0)} → {CASE_SCHEMA_VERSION})")
        if not dry_run:
            with write_transaction() as wconn:
                wconn.execute(
                    "UPDATE simulated_cases SET case_json=?, updated_at=? WHERE id=?",
                    (json.dumps(migrated), utcnow(), row["id"]),
                )
    if updated == 0:
        print(f"All {len(rows)} cases already at schema_version {CASE_SCHEMA_VERSION}.")
    else:
        verb = "would update" if dry_run else "updated"
        print(f"\n{verb} {updated}/{len(rows)} cases.")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="python -m backend.manage")
    sub = parser.add_subparsers(dest="command")

    p_migrate = sub.add_parser("migrate-cases",
                                help="Backfill case_json schema_version in the DB")
    p_migrate.add_argument("--dry-run", action="store_true",
                           help="Show what would change without writing")

    args = parser.parse_args(argv)
    if args.command == "migrate-cases":
        return cmd_migrate_cases(dry_run=args.dry_run)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
