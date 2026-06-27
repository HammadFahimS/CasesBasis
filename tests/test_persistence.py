from __future__ import annotations

import json

import pytest

from schemas.clinical_case import ClinicalCase
from schemas.doctor_consult import DoctorConsultResponse
from schemas.evaluation import ClinicalEvaluation, DiagnosisGuessEvaluation, InvestigationInterpretationFeedback, LearningRecommendations, QuestionRelevanceEvaluation, ReasoningFeedback
from schemas.investigation import InvestigationResult
from schemas.physical_exam import ExamFinding


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def db_path(tmp_path, monkeypatch):
    path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", path)
    from backend.db import run_migrations
    run_migrations()
    yield path


def make_case() -> ClinicalCase:
    return ClinicalCase.model_validate({
        "diagnosis": "myocardial infarction",
        "patient": {"age": 58, "gender": "male", "history": ["hypertension"]},
        "symptoms": ["chest pressure", "shortness of breath", "nausea"],
        "vitals": {"heart_rate": 104, "blood_pressure": "148/92", "respiratory_rate": 22, "oxygen_saturation": 95},
        "progression": [{"time_min": 0, "description": "Onset at rest."}],
        "risk_factors": ["hypertension"],
        "exam_findings": {"General": ["diaphoretic"]},
        "outcome": {"if_treated": "Good.", "if_untreated": "Bad."},
    })


def make_web_session(case: ClinicalCase):
    from backend.web_app import WebSession
    return WebSession(case=case)


def make_evaluation() -> ClinicalEvaluation:
    return ClinicalEvaluation(
        correct_diagnosis_guess=DiagnosisGuessEvaluation(
            provided=True, guess="myocardial infarction", is_correct=True, diagnosis_match_level="correct"
        ),
        key_symptoms_identified=["chest pressure"],
        key_symptoms_missed=[],
        relevance_of_questions=QuestionRelevanceEvaluation(
            relevant_count=3, irrelevant_count=0,
            relevant_questions=["chest pain?"], irrelevant_questions=[],
        ),
        overall_score=80,
        brief_feedback="Good work.",
        reasoning_feedback=ReasoningFeedback(
            diagnostic_path="systematic",
            question_order="appropriate",
            missed_reasoning_steps=[],
        ),
        learning_recommendations=LearningRecommendations(
            primary_skill_gap="history taking",
            focus_next="cover risk factors earlier",
            action_items=["ask about smoking"],
        ),
        investigation_interpretation_feedback=InvestigationInterpretationFeedback(),
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_migrations_create_tables(db_path):
    import sqlite3
    conn = sqlite3.connect(db_path)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    conn.close()
    assert "web_sessions" in tables
    assert "evaluations" in tables
    assert "events" in tables
    assert "residents" in tables
    assert "schema_migrations" in tables


def test_migrations_are_idempotent(db_path):
    from backend.db import run_migrations
    run_migrations()
    run_migrations()  # should not raise


def test_create_and_load_session_round_trip(db_path):
    from backend.persistence import create_session, load_session

    case = make_case()
    session = make_web_session(case)
    create_session("sess-1", "myocardial infarction", "medium", case)

    loaded = load_session("sess-1")
    assert loaded is not None
    assert loaded.case.diagnosis == "myocardial infarction"
    assert loaded.student_questions == []
    assert loaded.investigations == []


def test_load_session_returns_none_for_unknown_id(db_path):
    from backend.persistence import load_session

    assert load_session("does-not-exist") is None


def test_save_session_persists_state(db_path):
    from backend.persistence import create_session, load_session, save_session

    case = make_case()
    session = make_web_session(case)
    create_session("sess-2", "myocardial infarction", "medium", case)

    session.student_questions.append("Do you have chest pain?")
    session.patient_responses.append("Yes, a lot.")
    session.interpretations.append("Likely ischemic.")
    session.diagnosis_guess = "myocardial infarction"
    save_session("sess-2", session)

    loaded = load_session("sess-2")
    assert loaded is not None
    assert loaded.student_questions == ["Do you have chest pain?"]
    assert loaded.patient_responses == ["Yes, a lot."]
    assert loaded.interpretations == ["Likely ischemic."]
    assert loaded.diagnosis_guess == "myocardial infarction"


def test_save_session_persists_investigation_results(db_path):
    from backend.persistence import create_session, load_session, save_session
    from backend.investigations import build_troponin_result

    case = make_case()
    session = make_web_session(case)
    create_session("sess-3", "myocardial infarction", "medium", case)

    result = build_troponin_result(case)
    session.investigations.append(result)
    save_session("sess-3", session)

    loaded = load_session("sess-3")
    assert loaded is not None
    assert len(loaded.investigations) == 1
    assert loaded.investigations[0].investigation == "Troponin"
    assert loaded.investigations[0].structured_result["values"][0]["flag"] == "high"


def test_save_session_persists_doctor_consult(db_path):
    from backend.persistence import create_session, load_session, save_session

    case = make_case()
    session = make_web_session(case)
    create_session("sess-4", "myocardial infarction", "medium", case)

    consult = DoctorConsultResponse(
        urgency_level="critical",
        concise_advice="Activate cath lab immediately.",
        recommended_next_steps=["ECG", "troponin"],
        missing_information=[],
        safety_warnings=["Do not delay"],
        should_escalate=True,
    )
    session.doctor_consults.append(consult)
    save_session("sess-4", session)

    loaded = load_session("sess-4")
    assert loaded is not None
    assert len(loaded.doctor_consults) == 1
    assert loaded.doctor_consults[0].urgency_level == "critical"
    assert loaded.doctor_consults[0].should_escalate is True


def test_save_evaluation_writes_row_and_marks_session(db_path):
    import sqlite3
    from backend.persistence import create_session, save_evaluation

    case = make_case()
    create_session("sess-5", "myocardial infarction", "medium", case)
    evaluation = make_evaluation()
    save_evaluation("sess-5", evaluation)

    import sqlite3 as _sqlite3
    conn = _sqlite3.connect(db_path)
    conn.row_factory = _sqlite3.Row
    eval_row = conn.execute("SELECT overall_score FROM evaluations WHERE session_id='sess-5'").fetchone()
    session_row = conn.execute("SELECT status, evaluated_at FROM web_sessions WHERE id='sess-5'").fetchone()
    conn.close()

    assert eval_row is not None
    assert eval_row["overall_score"] == 80
    assert session_row["status"] == "evaluated"
    assert session_row["evaluated_at"] is not None


def test_log_event_writes_row(db_path):
    import sqlite3
    from backend.persistence import create_session, log_event

    case = make_case()
    create_session("sess-6", "myocardial infarction", "medium", case)
    log_event("sess-6", "question.asked", {"question": "Any chest pain?", "turn_index": 0})

    import sqlite3 as _sqlite3
    conn = _sqlite3.connect(db_path)
    row = conn.execute("SELECT event_type, payload FROM events WHERE session_id='sess-6'").fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "question.asked"
    payload = json.loads(row[1])
    assert payload["question"] == "Any chest pain?"


def test_get_session_loads_from_db_on_cache_miss(db_path, monkeypatch):
    from backend.persistence import create_session, save_session
    import backend.web_app as web_app

    case = make_case()
    session = make_web_session(case)
    session.student_questions.append("Any pain?")
    session.patient_responses.append("Yes.")
    create_session("sess-7", "myocardial infarction", "medium", case)
    save_session("sess-7", session)

    # Clear in-memory cache to simulate server restart
    monkeypatch.setattr(web_app, "SESSIONS", {})

    loaded = web_app.get_session("sess-7")
    assert loaded.student_questions == ["Any pain?"]
    assert loaded.patient_responses == ["Yes."]


def test_get_session_raises_404_for_unknown_id(db_path):
    from fastapi import HTTPException
    import backend.web_app as web_app

    with pytest.raises(HTTPException) as exc_info:
        web_app.get_session("totally-unknown")
    assert exc_info.value.status_code == 404
