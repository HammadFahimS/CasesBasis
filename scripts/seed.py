"""Insert seed residents and pre-baked evaluations for development/demo use.

Usage:
    python scripts/seed.py

Safe to run multiple times — uses INSERT OR IGNORE throughout.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.db import get_db, run_migrations, utcnow

_SEED_RESIDENTS = [
    ("alice@example.com", "Alice Chen"),
    ("bob@example.com", "Bob Patel"),
    ("carol@example.com", "Carol Okafor"),
]

_CASE_JSON = json.dumps({
    "diagnosis": "myocardial infarction",
    "patient": {"age": 58, "gender": "male", "history": ["hypertension", "current smoker", "high cholesterol"]},
    "symptoms": ["chest pressure", "shortness of breath", "nausea", "sweating", "left arm discomfort"],
    "vitals": {"heart_rate": 104, "blood_pressure": "148/92", "respiratory_rate": 22, "oxygen_saturation": 95},
    "progression": [
        {"time_min": 0, "description": "Pressure starts while walking upstairs."},
        {"time_min": 20, "description": "Pressure intensifies with sweating and nausea."},
        {"time_min": 45, "description": "Discomfort radiates toward the left arm."},
    ],
    "risk_factors": ["smoking", "hypertension", "high cholesterol"],
    "exam_findings": {"General": ["diaphoretic", "anxious"], "Neurological": ["no focal neurologic deficit"]},
    "outcome": {
        "if_treated": "Rapid emergency treatment improves outcomes.",
        "if_untreated": "Ongoing ischemia can lead to instability.",
    },
})

def _make_evaluation(score: int, missed: list[str]) -> str:
    return json.dumps({
        "correct_diagnosis_guess": {
            "provided": score >= 60,
            "guess": "myocardial infarction" if score >= 60 else None,
            "is_correct": score >= 60,
            "diagnosis_match_level": "correct" if score >= 60 else None,
        },
        "key_symptoms_identified": ["chest pressure", "shortness of breath"],
        "key_symptoms_missed": missed,
        "relevance_of_questions": {
            "relevant_count": 4,
            "irrelevant_count": 1 if score < 60 else 0,
            "relevant_questions": ["Any chest pain?", "Short of breath?"],
            "irrelevant_questions": ["What's your favourite show?"] if score < 60 else [],
        },
        "overall_score": score,
        "brief_feedback": f"Score: {score}.",
        "reasoning_feedback": {
            "diagnostic_path": "systematic" if score >= 70 else "scattered",
            "question_order": "started with chief complaint",
            "missed_reasoning_steps": (
                ["Ask about radiation of pain."] if score < 70 else []
            ),
        },
        "learning_recommendations": {
            "primary_skill_gap": "history taking",
            "focus_next": "cover risk factors earlier",
            "action_items": ["ask about smoking", "ask about family history"],
        },
        "investigation_interpretation_feedback": {
            "correctly_interpreted_findings": [],
            "missed_findings": [],
            "incorrect_interpretations": [],
            "summary": "No investigation interpretation provided.",
        },
    })


_SEED_EVALUATIONS: list[tuple[int, list[str]]] = [
    (42, ["nausea", "sweating", "left arm discomfort"]),
    (67, ["sweating", "left arm discomfort"]),
    (78, ["left arm discomfort"]),
    (83, []),
]


def _days_ago(n: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=n)).isoformat()


def main() -> None:
    run_migrations()
    conn = get_db()

    for email, display_name in _SEED_RESIDENTS:
        resident_id = str(uuid4())
        conn.execute(
            "INSERT OR IGNORE INTO residents (id, email, display_name, created_at, last_seen) VALUES (?, ?, ?, ?, ?)",
            (resident_id, email, display_name, _days_ago(30), _days_ago(1)),
        )
        # Fetch actual id (may have been inserted on a previous run)
        row = conn.execute("SELECT id FROM residents WHERE email=?", (email,)).fetchone()
        resident_id = row["id"]

        for i, (score, missed) in enumerate(_SEED_EVALUATIONS):
            session_id = str(uuid4())
            created = _days_ago(20 - i * 5)
            conn.execute(
                "INSERT OR IGNORE INTO web_sessions "
                "(id, resident_id, diagnosis, difficulty, status, case_json, state_json, created_at, updated_at, evaluated_at) "
                "VALUES (?, ?, ?, ?, 'evaluated', ?, '{}', ?, ?, ?)",
                (session_id, resident_id, "myocardial infarction", "medium",
                 _CASE_JSON, created, created, created),
            )
            eval_id = str(uuid4())
            conn.execute(
                "INSERT OR IGNORE INTO evaluations "
                "(id, session_id, resident_id, evaluation_json, overall_score, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (eval_id, session_id, resident_id, _make_evaluation(score, missed), score, created),
            )

    conn.commit()
    conn.close()
    print("Seed complete.")


if __name__ == "__main__":
    main()
