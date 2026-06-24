"""Durable storage for voice-driven simulation sessions.

A voice run is fully described by the engine snapshot plus the two transcript
channels and the diagnostics cards produced so far, so it can be parked to the
DB and rehydrated later (full visual resume). Writes are best-effort: a storage
hiccup must never break a live turn, so failures are logged, not raised.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import TYPE_CHECKING, Any

from backend.db import get_db, utcnow, write_transaction

if TYPE_CHECKING:
    from backend.voice_session import VoiceSession

logger = logging.getLogger(__name__)


def save(session: "VoiceSession", status: str = "active") -> None:
    """Insert-or-update the parked state of a session (write-through, best-effort)."""
    now = utcnow()
    ended_at = now if status == "ended" else None
    try:
        with write_transaction() as conn:
            conn.execute(
                "INSERT INTO voice_sessions "
                "(id, resident_id, case_id, status, snapshot_json, history_json, "
                " patient_history_json, cards_json, created_at, updated_at, ended_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "  status=excluded.status, snapshot_json=excluded.snapshot_json, "
                "  history_json=excluded.history_json, "
                "  patient_history_json=excluded.patient_history_json, "
                "  cards_json=excluded.cards_json, updated_at=excluded.updated_at, "
                "  ended_at=COALESCE(excluded.ended_at, voice_sessions.ended_at)",
                (
                    session.id, session.resident_id, session.case.id, status,
                    json.dumps(session.engine.snapshot()),
                    json.dumps(session.history),
                    json.dumps(session.patient_history),
                    json.dumps(session.cards),
                    now, now, ended_at,
                ),
            )
    except sqlite3.Error as exc:
        logger.warning("voice session save failed for %s: %s", session.id, exc)


def load(session_id: str) -> dict[str, Any] | None:
    """Return the raw parked row (decoded JSON) for a session, or None."""
    try:
        row = get_db().execute(
            "SELECT id, resident_id, case_id, status, snapshot_json, history_json, "
            "       patient_history_json, cards_json FROM voice_sessions WHERE id=?",
            (session_id,),
        ).fetchone()
    except sqlite3.Error as exc:
        logger.warning("voice session load failed for %s: %s", session_id, exc)
        return None
    if row is None:
        return None
    return {
        "id": row[0],
        "resident_id": row[1],
        "case_id": row[2],
        "status": row[3],
        "snapshot": json.loads(row[4] or "{}"),
        "history": json.loads(row[5] or "[]"),
        "patient_history": json.loads(row[6] or "[]"),
        "cards": json.loads(row[7] or "[]"),
    }


def mark_ended(session_id: str) -> None:
    """Flag a session as concluded (idempotent, best-effort)."""
    try:
        with write_transaction() as conn:
            conn.execute(
                "UPDATE voice_sessions SET status='ended', ended_at=?, updated_at=? WHERE id=?",
                (utcnow(), utcnow(), session_id),
            )
    except sqlite3.Error as exc:
        logger.warning("voice session mark_ended failed for %s: %s", session_id, exc)


def delete(session_id: str, resident_id: str) -> bool:
    """Remove a run, scoped to its owner. Returns True if a row was deleted."""
    try:
        with write_transaction() as conn:
            cur = conn.execute(
                "DELETE FROM voice_sessions WHERE id=? AND resident_id=?",
                (session_id, resident_id),
            )
            return cur.rowcount > 0
    except sqlite3.Error as exc:
        logger.warning("voice session delete failed for %s: %s", session_id, exc)
        return False


def record_evaluation(
    session_id: str, score: int, assessment: list[dict[str, Any]], duration_seconds: int,
) -> None:
    """Persist a concluded run's score so it appears in analytics. Best-effort and
    idempotent — re-viewing a debrief simply rewrites the same figures."""
    now = utcnow()
    try:
        with write_transaction() as conn:
            conn.execute(
                "UPDATE voice_sessions SET overall_score=?, assessment_json=?, "
                "duration_seconds=?, evaluated_at=?, status='ended', "
                "ended_at=COALESCE(ended_at, ?), updated_at=? WHERE id=?",
                (score, json.dumps(assessment), max(0, int(duration_seconds)),
                 now, now, now, session_id),
            )
    except sqlite3.Error as exc:
        logger.warning("voice evaluation record failed for %s: %s", session_id, exc)


def _case_meta(case_id: str) -> tuple[str, str]:
    """(title, difficulty) for a case id — for the analytics row label. Fail-soft."""
    try:
        from backend.simulated_case_store import load_seed_case
        meta = load_seed_case(case_id).meta
        return (meta.title or case_id, meta.difficulty or "medium")
    except Exception:
        return (case_id, "medium")


def get_resident_evaluations(resident_id: str, limit: int = 200) -> list[dict[str, Any]]:
    """Concluded, scored voice runs shaped like the dashboard's session rows, so
    they merge straight into the analytics list and charts (tagged source=voice)."""
    try:
        rows = get_db().execute(
            "SELECT id, case_id, created_at, duration_seconds, overall_score "
            "FROM voice_sessions WHERE resident_id=? AND overall_score IS NOT NULL "
            "ORDER BY created_at DESC LIMIT ?",
            (resident_id, limit),
        ).fetchall()
    except sqlite3.Error as exc:
        logger.warning("voice evaluations read failed for %s: %s", resident_id, exc)
        return []
    out = []
    for r in rows:
        title, difficulty = _case_meta(r[1])
        out.append({
            "id": r[0], "diagnosis": title, "difficulty": difficulty,
            "status": "evaluated", "created_at": r[2],
            "duration_seconds": r[3] or 0, "overall_score": r[4], "source": "voice",
        })
    return out


def get_evaluation(session_id: str, resident_id: str) -> dict[str, Any] | None:
    """The stored assessment for one voice run, for the analytics review drawer."""
    try:
        row = get_db().execute(
            "SELECT case_id, created_at, duration_seconds, overall_score, assessment_json, evaluated_at "
            "FROM voice_sessions WHERE id=? AND resident_id=? AND overall_score IS NOT NULL",
            (session_id, resident_id),
        ).fetchone()
    except sqlite3.Error as exc:
        logger.warning("voice evaluation detail failed for %s: %s", session_id, exc)
        return None
    if row is None:
        return None
    title, difficulty = _case_meta(row[0])
    return {
        "id": session_id, "diagnosis": title, "difficulty": difficulty,
        "created_at": row[1], "duration_seconds": row[2] or 0,
        "overall_score": row[3], "assessment": json.loads(row[4] or "[]"),
        "evaluated_at": row[5], "source": "voice",
    }


def list_for_resident(resident_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """Resumable/recent runs for a resident (newest first), for the home surface."""
    try:
        rows = get_db().execute(
            "SELECT id, case_id, status, created_at, updated_at, ended_at "
            "FROM voice_sessions WHERE resident_id=? "
            "ORDER BY updated_at DESC LIMIT ?",
            (resident_id, limit),
        ).fetchall()
    except sqlite3.Error as exc:
        logger.warning("voice session list failed for %s: %s", resident_id, exc)
        return []
    return [
        {"id": r[0], "case_id": r[1], "status": r[2],
         "created_at": r[3], "updated_at": r[4], "ended_at": r[5]}
        for r in rows
    ]
