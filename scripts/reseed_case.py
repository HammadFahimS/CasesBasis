"""Re-seed a gold simulated case's DB row from its ``cases/*.json`` file.

The live voice sim reads each case from the **database** (``get_case``), never from
the file. Startup seeding (``seed_gold_cases``) is insert-only and never overwrites
an existing row, so edits to ``cases/<id>.json`` are invisible to the running app
until the row is re-seeded. This script closes that gap.

Usage:
    python scripts/reseed_case.py myasthenic_crisis     # one case
    python scripts/reseed_case.py --all                 # every authored gold case
    python scripts/reseed_case.py --list                # show authored case ids

Notes:
- Validates the JSON against the schema before writing; a malformed file aborts
  without touching the DB.
- No server restart is needed (the DB is read per new session), but a parked or
  resumed session keeps its old case — start a *fresh* session to see the change.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.db import run_migrations
from backend.simulated_case_repo import reseed_gold_case, seed_gold_cases
from backend.simulated_case_store import list_seed_cases


def _authored_ids() -> list[str]:
    return [str(s["id"]) for s in list_seed_cases()]


def main(argv: list[str]) -> int:
    run_migrations()
    seed_gold_cases()  # ensure rows exist before we update them

    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__)
        return 0

    if argv[0] == "--list":
        ids = _authored_ids()
        print("Authored gold cases:" if ids else "No authored cases found.")
        for cid in ids:
            print(f"  {cid}")
        return 0

    case_ids = _authored_ids() if argv[0] == "--all" else argv

    failures = 0
    for case_id in case_ids:
        try:
            if reseed_gold_case(case_id):
                print(f"✓ reseeded {case_id}")
            else:
                print(f"✗ {case_id}: no DB row (not a seeded gold case?)")
                failures += 1
        except FileNotFoundError:
            print(f"✗ {case_id}: no cases/{case_id}.json file")
            failures += 1
        except Exception as exc:  # validation or DB error — report, don't crash the batch
            print(f"✗ {case_id}: {type(exc).__name__}: {exc}")
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
