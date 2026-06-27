"""Database repository for simulated dynamic cases.

The ``simulated_cases`` table is the source of truth at runtime. Authored gold
cases under ``cases/*.json`` are seeded into the table once on startup and
flagged ``is_gold`` (protected: no edit/delete, copy only).

Each row stores the full :class:`schemas.simulated_case.SimulatedCase` document
(``case_json``) plus editor node positions (``layout_json``) kept separate so
the clinical schema stays presentation-free.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import ValidationError

from backend.db import get_db, utcnow
from backend.simulated_case_store import list_seed_cases, load_seed_case
from schemas.simulated_case import SimulatedCase

logger = logging.getLogger(__name__)


class CaseProtectedError(PermissionError):
    """Raised when an edit/delete is attempted on a protected gold case."""


class CaseValidationError(ValueError):
    """Raised when a submitted case document fails schema validation."""


def seed_gold_cases() -> None:
    """Insert authored ``cases/*.json`` documents as protected gold cases.

    Idempotent: only inserts ids not already present, and never overwrites an
    existing row (so the seed file can't clobber later in-DB edits if a future
    case is un-golded). Safe to call on every startup.
    """
    conn = get_db()
    existing = {row[0] for row in conn.execute("SELECT id FROM simulated_cases")}
    for summary in list_seed_cases():
        case_id = str(summary["id"])
        if case_id in existing:
            continue
        try:
            case = load_seed_case(case_id)
        except Exception as exc:  # malformed seed shouldn't break startup
            logger.warning("skipping seed case %s: %s", case_id, exc)
            continue
        now = utcnow()
        conn.execute(
            "INSERT INTO simulated_cases "
            "(id, title, specialty, difficulty, case_json, layout_json, is_gold, created_by, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, '{}', 1, NULL, ?, ?)",
            (case.id, case.meta.title, case.meta.specialty, case.meta.difficulty,
             case.model_dump_json(), now, now),
        )
        logger.info("seeded gold simulated case: %s", case_id)
    conn.commit()


def reseed_gold_case(case_id: str) -> bool:
    """Overwrite an existing gold case's DB row from its ``cases/*.json`` file.

    The deliberate counterpart to :func:`seed_gold_cases` (which is insert-only and
    never clobbers a row). Use this after editing a gold case JSON so the change
    reaches the live sim, which reads the case from the DB via :func:`get_case` —
    never from the file. ``is_gold`` and ``created_at`` are preserved; only the
    case content and ``updated_at`` change. Returns ``False`` if the row is absent
    (call ``seed_gold_cases`` first) and raises if the JSON fails validation.

    No server restart is needed (the DB is read per new session), but a parked or
    resumed session keeps its old case — start a *fresh* session to see the change.
    """
    case = load_seed_case(case_id)  # validates against the schema; raises on bad JSON
    conn = get_db()
    row = conn.execute(
        "SELECT is_gold FROM simulated_cases WHERE id=?", (case.id,)
    ).fetchone()
    if row is None:
        return False
    now = utcnow()
    conn.execute(
        "UPDATE simulated_cases SET title=?, specialty=?, difficulty=?, "
        "case_json=?, updated_at=? WHERE id=?",
        (case.meta.title, case.meta.specialty, case.meta.difficulty,
         case.model_dump_json(), now, case.id),
    )
    conn.commit()
    logger.info("reseeded gold simulated case from file: %s", case_id)
    return True


def list_cases() -> list[dict[str, Any]]:
    """Catalogue rows for the CRUD table."""
    conn = get_db()
    rows = conn.execute(
        "SELECT id, title, specialty, difficulty, is_gold, created_at, updated_at, case_json "
        "FROM simulated_cases ORDER BY is_gold DESC, updated_at DESC"
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        doc = json.loads(row["case_json"])
        out.append({
            "id": row["id"],
            "title": row["title"],
            "specialty": row["specialty"],
            "difficulty": row["difficulty"],
            "is_gold": bool(row["is_gold"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "state_count": len(doc.get("states", [])),
            "summary": doc.get("meta", {}).get("summary", ""),
        })
    return out


def get_case(case_id: str) -> dict[str, Any] | None:
    """Return the full case + layout + protection flag, or None.

    Applies ``migrate_case_json`` on every read so the in-memory dict is always
    at the current schema version without requiring a DB write-back."""
    conn = get_db()
    row = conn.execute(
        "SELECT id, case_json, layout_json, is_gold, created_at, updated_at "
        "FROM simulated_cases WHERE id=?",
        (case_id,),
    ).fetchone()
    if row is None:
        return None
    from backend.case_migrations import migrate_case_json
    case_raw = migrate_case_json(json.loads(row["case_json"]))
    return {
        "id": row["id"],
        "case": case_raw,
        "layout": json.loads(row["layout_json"] or "{}"),
        "is_gold": bool(row["is_gold"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _validate(case_payload: dict[str, Any]) -> SimulatedCase:
    try:
        return SimulatedCase.model_validate(case_payload)
    except ValidationError as exc:
        raise CaseValidationError(str(exc)) from exc


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or "case"


def _unique_id(base: str) -> str:
    conn = get_db()
    existing = {row[0] for row in conn.execute("SELECT id FROM simulated_cases")}
    if base not in existing:
        return base
    n = 2
    while f"{base}_{n}" in existing:
        n += 1
    return f"{base}_{n}"


def create_case(
    case_payload: dict[str, Any],
    layout: dict[str, Any] | None = None,
    created_by: str | None = None,
) -> dict[str, Any]:
    """Validate and insert a new (non-gold) case. The id is derived from the
    document id (or its title) and made unique."""
    case = _validate(case_payload)
    case_id = _unique_id(_slugify(case.id or case.meta.title))
    # Keep the stored document id in sync with the row id.
    payload = case.model_dump()
    payload["id"] = case_id
    case = _validate(payload)

    now = utcnow()
    conn = get_db()
    conn.execute(
        "INSERT INTO simulated_cases "
        "(id, title, specialty, difficulty, case_json, layout_json, is_gold, created_by, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?)",
        (case_id, case.meta.title, case.meta.specialty, case.meta.difficulty,
         case.model_dump_json(), json.dumps(layout or {}), created_by, now, now),
    )
    conn.commit()
    return get_case(case_id)  # type: ignore[return-value]


def update_case(
    case_id: str,
    case_payload: dict[str, Any],
    layout: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and overwrite an existing case. Rejects gold cases."""
    existing = get_case(case_id)
    if existing is None:
        raise KeyError(case_id)
    if existing["is_gold"]:
        raise CaseProtectedError(case_id)

    payload = dict(case_payload)
    payload["id"] = case_id  # id is immutable on update
    case = _validate(payload)

    conn = get_db()
    conn.execute(
        "UPDATE simulated_cases SET title=?, specialty=?, difficulty=?, case_json=?, layout_json=?, updated_at=? "
        "WHERE id=?",
        (case.meta.title, case.meta.specialty, case.meta.difficulty, case.model_dump_json(),
         json.dumps(layout if layout is not None else existing["layout"]), utcnow(), case_id),
    )
    conn.commit()
    return get_case(case_id)  # type: ignore[return-value]


def delete_case(case_id: str) -> None:
    """Delete a case. Rejects gold cases."""
    existing = get_case(case_id)
    if existing is None:
        raise KeyError(case_id)
    if existing["is_gold"]:
        raise CaseProtectedError(case_id)
    conn = get_db()
    conn.execute("DELETE FROM simulated_cases WHERE id=?", (case_id,))
    conn.commit()


def duplicate_case(case_id: str, created_by: str | None = None) -> dict[str, Any]:
    """Copy any case (including gold) into a new editable, non-gold case."""
    src = get_case(case_id)
    if src is None:
        raise KeyError(case_id)
    payload = dict(src["case"])
    new_id = _unique_id(f"{_slugify(payload.get('id') or 'case')}_copy")
    payload["id"] = new_id
    meta = dict(payload.get("meta", {}))
    meta["title"] = f"{meta.get('title', 'Untitled')} (copy)"
    payload["meta"] = meta
    return create_case(payload, layout=src["layout"], created_by=created_by)
