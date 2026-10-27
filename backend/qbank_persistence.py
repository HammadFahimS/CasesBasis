"""Durable per-resident Question Bank progress.

One row per answered checkpoint (``qbank_attempts``) is the UWorld-style history:
it lets a resident see cumulative performance overall and broken down by subject
and system, and powers per-checkpoint "seen before / got it right last time"
lookups. Writes are best-effort — a storage hiccup must never break a practice
session — so failures are logged, not raised (mirrors ``voice_persistence``).
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from typing import Any

from backend.db import get_db, utcnow, write_transaction

logger = logging.getLogger(__name__)


def record_attempt(
    resident_id: str,
    case_id: str,
    checkpoint_id: str,
    correct: bool,
    chosen_index: int | None,
    mode: str = "tutor",
    subject: str = "",
    system: str = "",
) -> None:
    """Append one answered-checkpoint attempt (best-effort)."""
    try:
        with write_transaction() as conn:
            conn.execute(
                "INSERT INTO qbank_attempts "
                "(id, resident_id, case_id, checkpoint_id, correct, chosen_index, "
                " mode, subject, system, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    uuid.uuid4().hex, resident_id, case_id, checkpoint_id,
                    1 if correct else 0, chosen_index, mode or "tutor",
                    subject or "", system or "", utcnow(),
                ),
            )
    except sqlite3.Error as exc:
        logger.warning("qbank attempt record failed for %s: %s", resident_id, exc)


def _breakdown(conn, resident_id: str, column: str) -> list[dict[str, Any]]:
    try:
        rows = conn.execute(
            f"SELECT {column} AS k, COUNT(*) AS total, SUM(correct) AS got "
            f"FROM qbank_attempts WHERE resident_id=? AND {column} != '' "
            f"GROUP BY {column} ORDER BY {column}",
            (resident_id,),
        ).fetchall()
    except sqlite3.Error as exc:
        logger.warning("qbank breakdown failed (%s) for %s: %s", column, resident_id, exc)
        return []
    out = []
    for r in rows:
        total = int(r[1] or 0)
        got = int(r[2] or 0)
        out.append({
            "key": r[0], "total": total, "correct": got,
            "percent": round(100.0 * got / total) if total else 0,
        })
    return out


def stats(resident_id: str) -> dict[str, Any]:
    """Cumulative performance for a resident: overall + by subject and system.

    ``total`` counts every attempt; ``answered_checkpoints`` counts distinct
    checkpoints seen (UWorld's "used" figure). Fail-soft: returns zeros on error.
    """
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS total, COALESCE(SUM(correct), 0) AS got, "
            "       COUNT(DISTINCT checkpoint_id) AS seen "
            "FROM qbank_attempts WHERE resident_id=?",
            (resident_id,),
        ).fetchone()
    except sqlite3.Error as exc:
        logger.warning("qbank stats failed for %s: %s", resident_id, exc)
        row = None
    total = int(row[0] or 0) if row else 0
    got = int(row[1] or 0) if row else 0
    seen = int(row[2] or 0) if row else 0
    return {
        "total": total,
        "correct": got,
        "percent": round(100.0 * got / total) if total else 0,
        "answered_checkpoints": seen,
        "by_subject": _breakdown(conn, resident_id, "subject"),
        "by_system": _breakdown(conn, resident_id, "system"),
    }


def answered_checkpoint_ids(resident_id: str) -> set[str]:
    """Checkpoints this resident has already answered (for "seen before" hints)."""
    try:
        rows = get_db().execute(
            "SELECT DISTINCT checkpoint_id FROM qbank_attempts WHERE resident_id=?",
            (resident_id,),
        ).fetchall()
    except sqlite3.Error as exc:
        logger.warning("qbank answered-ids read failed for %s: %s", resident_id, exc)
        return set()
    return {r[0] for r in rows}
