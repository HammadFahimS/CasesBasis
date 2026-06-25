from __future__ import annotations

import json
import logging
import sqlite3
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from backend.db import get_db, utcnow, write_transaction

logger = logging.getLogger(__name__)

from schemas.account import ResidentAccount
from schemas.clinical_case import ClinicalCase
from schemas.doctor_consult import DoctorConsultResponse
from schemas.evaluation import ClinicalEvaluation
from schemas.investigation import InvestigationResult
from schemas.physical_exam import ExamFinding

if TYPE_CHECKING:
    from backend.sim_core import WebSession


def create_session(
    session_id: str,
    diagnosis: str,
    difficulty: str,
    case: ClinicalCase,
    resident_id: str | None = None,
) -> None:
    now = utcnow()
    conn = get_db()
    conn.execute(
        "INSERT INTO web_sessions "
        "(id, resident_id, diagnosis, difficulty, status, case_json, state_json, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, 'active', ?, '{}', ?, ?)",
        (session_id, resident_id, diagnosis, difficulty, case.model_dump_json(), now, now),
    )
    conn.commit()


def save_session(session_id: str, session: WebSession) -> None:
    try:
        conn = get_db()
        conn.execute(
            "UPDATE web_sessions SET state_json=?, updated_at=? WHERE id=?",
            (json.dumps(_serialize_state(session)), utcnow(), session_id),
        )
        conn.commit()
    except sqlite3.Error as exc:
        logger.warning("save_session failed for %s: %s", session_id, exc)


def load_session(session_id: str) -> WebSession | None:
    from backend.sim_core import WebSession

    conn = get_db()
    row = conn.execute(
        "SELECT case_json, state_json FROM web_sessions WHERE id=?",
        (session_id,),
    ).fetchone()
    if row is None:
        return None
    case = ClinicalCase.model_validate_json(row["case_json"])
    state = json.loads(row["state_json"] or "{}")
    return _deserialize_state(case, state)


def save_evaluation(
    session_id: str,
    evaluation: ClinicalEvaluation,
    resident_id: str | None = None,
    elapsed_seconds: int = 0,
) -> None:
    try:
        evaluation_id = str(uuid4())
        now = utcnow()
        with write_transaction() as conn:
            conn.execute(
                "INSERT INTO evaluations (id, session_id, resident_id, evaluation_json, overall_score, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    evaluation_id,
                    session_id,
                    resident_id,
                    evaluation.model_dump_json(),
                    evaluation.overall_score,
                    now,
                ),
            )
            conn.execute(
                "UPDATE web_sessions SET status='evaluated', evaluated_at=?, updated_at=?, "
                "duration_seconds = duration_seconds + ? WHERE id=?",
                (now, now, elapsed_seconds, session_id),
            )
    except Exception as exc:
        logger.warning("save_evaluation failed for %s: %s", session_id, exc)


def log_event(
    session_id: str,
    event_type: str,
    payload: dict[str, Any] | None = None,
    resident_id: str | None = None,
) -> None:
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO events (session_id, resident_id, event_type, payload, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, resident_id, event_type, json.dumps(payload or {}), utcnow()),
        )
        conn.commit()
    except sqlite3.Error as exc:
        logger.warning("log_event failed for %s/%s: %s", session_id, event_type, exc)


def _serialize_state(session: WebSession) -> dict[str, Any]:
    return {
        "student_questions": session.student_questions,
        "patient_responses": session.patient_responses,
        "investigations": [r.model_dump(mode="json") for r in session.investigations],
        "interpretations": session.interpretations,
        "doctor_consults": [c.model_dump(mode="json") for c in session.doctor_consults],
        "diagnosis_guess": session.diagnosis_guess,
        "physical_exams": [e.model_dump(mode="json") for e in session.physical_exams],
        "revealed_symptoms": session.revealed_symptoms,
    }


def _deserialize_state(case: ClinicalCase, state: dict[str, Any]) -> WebSession:
    from backend.sim_core import WebSession

    return WebSession(
        case=case,
        student_questions=state.get("student_questions", []),
        patient_responses=state.get("patient_responses", []),
        investigations=[
            InvestigationResult.model_validate(r) for r in state.get("investigations", [])
        ],
        interpretations=state.get("interpretations", []),
        doctor_consults=[
            DoctorConsultResponse.model_validate(c) for c in state.get("doctor_consults", [])
        ],
        diagnosis_guess=state.get("diagnosis_guess"),
        physical_exams=[ExamFinding.model_validate(e) for e in state.get("physical_exams", [])],
        revealed_symptoms=state.get("revealed_symptoms", []),
    )


# ── Resident helpers ──────────────────────────────────────────────────────────

def find_resident_by_code(resident_code: str, last_name: str) -> str | None:
    with write_transaction() as conn:
        row = conn.execute(
            "SELECT id FROM residents WHERE resident_code=? AND lower(last_name)=lower(?)",
            (resident_code.upper(), last_name.strip()),
        ).fetchone()
        if row is None:
            return None
        conn.execute("UPDATE residents SET last_seen=? WHERE id=?", (utcnow(), row["id"]))
    return row["id"]


def create_resident_with_code(resident_code: str, last_name: str) -> str:
    resident_id = str(uuid4())
    now = utcnow()
    display_name = last_name.strip().title()
    conn = get_db()
    conn.execute(
        "INSERT INTO residents (id, email, display_name, resident_code, last_name, created_at, last_seen) "
        "VALUES (?, NULL, ?, ?, ?, ?, ?)",
        (resident_id, display_name, resident_code.upper(), last_name.strip(), now, now),
    )
    conn.commit()
    return resident_id


def get_or_create_resident(email: str) -> str:
    email = email.strip().lower()
    now = utcnow()
    with write_transaction() as conn:
        row = conn.execute("SELECT id FROM residents WHERE email=?", (email,)).fetchone()
        if row:
            conn.execute("UPDATE residents SET last_seen=? WHERE id=?", (now, row["id"]))
            return row["id"]
        resident_id = str(uuid4())
        display_name = email.split("@")[0].replace(".", " ").replace("_", " ").title()
        conn.execute(
            "INSERT INTO residents (id, email, display_name, created_at, last_seen) VALUES (?, ?, ?, ?, ?)",
            (resident_id, email, display_name, now, now),
        )
    return resident_id


def get_resident(resident_id: str) -> ResidentAccount | None:
    conn = get_db()
    row = conn.execute(
        "SELECT id, email, display_name, created_at, last_seen FROM residents WHERE id=?",
        (resident_id,),
    ).fetchone()
    if row is None:
        return None
    return ResidentAccount(
        id=row["id"],
        email=row["email"],
        display_name=row["display_name"],
        created_at=row["created_at"],
        last_seen=row["last_seen"],
    )


def store_magic_link(token: str, resident_id: str, expires_at: str) -> None:
    conn = get_db()
    conn.execute(
        "INSERT INTO magic_links (token, resident_id, expires_at) VALUES (?, ?, ?)",
        (token, resident_id, expires_at),
    )
    conn.commit()


def consume_magic_link(token: str) -> str | None:
    now = utcnow()
    with write_transaction() as conn:
        row = conn.execute(
            "SELECT resident_id, expires_at, used FROM magic_links WHERE token=?",
            (token,),
        ).fetchone()
        if row is None or row["used"] or row["expires_at"] < now:
            return None
        conn.execute("UPDATE magic_links SET used=1 WHERE token=?", (token,))
    return row["resident_id"]


# ── Dashboard helpers ─────────────────────────────────────────────────────────

def get_resident_sessions(resident_id: str, limit: int = 200) -> list[dict[str, Any]]:
    conn = get_db()
    rows = conn.execute(
        "SELECT ws.id, ws.diagnosis, ws.difficulty, ws.status, ws.created_at, ws.duration_seconds, "
        "(SELECT overall_score FROM evaluations WHERE session_id = ws.id "
        " ORDER BY created_at DESC LIMIT 1) AS overall_score "
        "FROM web_sessions ws "
        "WHERE ws.resident_id = ? "
        "ORDER BY ws.created_at DESC LIMIT ?",
        (resident_id, limit),
    ).fetchall()
    return [dict(row) for row in rows]


def get_resident_stats(resident_id: str) -> dict[str, Any]:
    conn = get_db()
    rows = conn.execute(
        "SELECT overall_score, evaluation_json FROM evaluations "
        "WHERE resident_id=? ORDER BY created_at DESC LIMIT 20",
        (resident_id,),
    ).fetchall()
    dur_row = conn.execute(
        "SELECT AVG(duration_seconds) as avg_dur FROM web_sessions "
        "WHERE resident_id=? AND status='evaluated' AND duration_seconds > 0",
        (resident_id,),
    ).fetchone()

    avg_duration = int(round(dur_row["avg_dur"])) if dur_row and dur_row["avg_dur"] else None

    if not rows:
        return {
            "avg_score": None,
            "cases_completed": 0,
            "common_missed_symptoms": [],
            "recommended_difficulty": "medium",
            "avg_duration_seconds": avg_duration,
        }

    scores = [row["overall_score"] for row in rows]
    avg: float = round(sum(scores) / len(scores), 2)

    missed_counter: dict[str, int] = {}
    for row in rows:
        try:
            eval_data = json.loads(row["evaluation_json"])
            for symptom in eval_data.get("key_symptoms_missed", []):
                missed_counter[symptom] = missed_counter.get(symptom, 0) + 1
        except Exception:
            pass
    common_missed = sorted(missed_counter, key=lambda s: -missed_counter[s])[:5]

    from backend.performance_tracker import recommend_difficulty
    return {
        "avg_score": avg,
        "cases_completed": len(rows),
        "common_missed_symptoms": common_missed,
        "recommended_difficulty": recommend_difficulty(avg),
        "avg_duration_seconds": avg_duration,
    }


def get_session_for_resident(session_id: str, resident_id: str) -> dict[str, Any] | None:
    conn = get_db()
    row = conn.execute(
        "SELECT id, diagnosis, difficulty, status, case_json, state_json, created_at, graph_snapshot, duration_seconds "
        "FROM web_sessions WHERE id=? AND resident_id=?",
        (session_id, resident_id),
    ).fetchone()
    return dict(row) if row else None


def delete_session_for_resident(session_id: str, resident_id: str) -> bool:
    conn = get_db()
    cursor = conn.execute(
        "DELETE FROM web_sessions WHERE id=? AND resident_id=?",
        (session_id, resident_id),
    )
    conn.commit()
    return cursor.rowcount > 0


def get_session_evaluation(session_id: str) -> dict[str, Any] | None:
    conn = get_db()
    row = conn.execute(
        "SELECT evaluation_json, overall_score, created_at FROM evaluations "
        "WHERE session_id=? ORDER BY created_at DESC LIMIT 1",
        (session_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "evaluation": json.loads(row["evaluation_json"]),
        "overall_score": row["overall_score"],
        "evaluated_at": row["created_at"],
    }


def pause_session(session_id: str, graph_snapshot: str | None = None, elapsed_seconds: int = 0) -> None:
    conn = get_db()
    conn.execute(
        "UPDATE web_sessions SET status='paused', graph_snapshot=?, updated_at=?, "
        "duration_seconds = duration_seconds + ? WHERE id=?",
        (graph_snapshot, utcnow(), elapsed_seconds, session_id),
    )
    conn.commit()


def get_recent_evaluations_for_coaching(resident_id: str, limit: int = 5) -> list[dict[str, Any]]:
    conn = get_db()
    rows = conn.execute(
        "SELECT e.evaluation_json, e.overall_score, ws.diagnosis "
        "FROM evaluations e JOIN web_sessions ws ON e.session_id = ws.id "
        "WHERE e.resident_id=? ORDER BY e.created_at DESC LIMIT ?",
        (resident_id, limit),
    ).fetchall()
    result = []
    for row in rows:
        try:
            result.append({
                "diagnosis": row["diagnosis"],
                "overall_score": row["overall_score"],
                "evaluation": json.loads(row["evaluation_json"]),
            })
        except Exception:
            pass
    return result


def get_session_duration(session_id: str) -> int:
    conn = get_db()
    row = conn.execute(
        "SELECT duration_seconds FROM web_sessions WHERE id=?", (session_id,)
    ).fetchone()
    return row["duration_seconds"] if row else 0
