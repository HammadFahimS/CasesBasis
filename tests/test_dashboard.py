from __future__ import annotations

import json

import pytest

from schemas.clinical_case import ClinicalCase
from schemas.doctor_consult import DoctorConsultResponse
from schemas.evaluation import (
    ClinicalEvaluation,
    DiagnosisGuessEvaluation,
    InvestigationInterpretationFeedback,
    LearningRecommendations,
    QuestionRelevanceEvaluation,
    ReasoningFeedback,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def db_path(tmp_path, monkeypatch):
    path = str(tmp_path / "dash_test.db")
    monkeypatch.setenv("DB_PATH", path)
    monkeypatch.setenv("APP_SECRET", "test-secret-dashboard")
    from backend.db import run_migrations
    run_migrations()
    yield path



def make_case() -> ClinicalCase:
    return ClinicalCase.model_validate({
        "diagnosis": "myocardial infarction",
        "patient": {"age": 58, "gender": "male", "history": ["hypertension"]},
        "symptoms": ["chest pressure", "shortness of breath", "nausea"],
        "vitals": {"heart_rate": 104, "blood_pressure": "148/92", "respiratory_rate": 22, "oxygen_saturation": 95},
        "progression": [{"time_min": 0, "description": "Onset."}],
        "risk_factors": ["hypertension"],
        "exam_findings": {"General": ["diaphoretic"]},
        "outcome": {"if_treated": "Good.", "if_untreated": "Bad."},
    })


def make_web_session(case: ClinicalCase):
    from backend.web_app import WebSession
    return WebSession(case=case)


def make_evaluation(score: int, missed: list[str] | None = None) -> ClinicalEvaluation:
    return ClinicalEvaluation(
        correct_diagnosis_guess=DiagnosisGuessEvaluation(
            provided=True, guess="mi", is_correct=True, diagnosis_match_level="correct"
        ),
        key_symptoms_identified=["chest pressure"],
        key_symptoms_missed=missed or [],
        relevance_of_questions=QuestionRelevanceEvaluation(
            relevant_count=3, irrelevant_count=0,
            relevant_questions=["chest pain?"], irrelevant_questions=[],
        ),
        overall_score=score,
        brief_feedback="ok",
        reasoning_feedback=ReasoningFeedback(
            diagnostic_path="systematic", question_order="good", missed_reasoning_steps=[],
        ),
        learning_recommendations=LearningRecommendations(
            primary_skill_gap="history", focus_next="risk factors", action_items=["ask smoking"],
        ),
        investigation_interpretation_feedback=InvestigationInterpretationFeedback(),
    )


def make_authed_request(resident_id: str) -> object:
    from backend.auth import COOKIE_NAME, make_session_cookie
    cookie_value = make_session_cookie(resident_id)

    class FakeRequest:
        cookies = {COOKIE_NAME: cookie_value}

    return FakeRequest()


def make_anon_request() -> object:
    class FakeRequest:
        cookies: dict = {}

    return FakeRequest()


# ── Persistence helpers ───────────────────────────────────────────────────────

def test_get_resident_sessions_empty(db_path):
    from backend.persistence import get_or_create_resident, get_resident_sessions

    resident_id = get_or_create_resident("alice@example.com")
    sessions = get_resident_sessions(resident_id)
    assert sessions == []


def test_get_resident_sessions_returns_own_sessions_only(db_path):
    from backend.persistence import (
        create_session,
        get_or_create_resident,
        get_resident_sessions,
        save_evaluation,
    )

    alice_id = get_or_create_resident("alice@example.com")
    bob_id = get_or_create_resident("bob@example.com")
    case = make_case()

    create_session("sess-a", "mi", "medium", case, resident_id=alice_id)
    create_session("sess-b", "mi", "medium", case, resident_id=bob_id)
    save_evaluation("sess-a", make_evaluation(74), resident_id=alice_id)

    sessions = get_resident_sessions(alice_id)
    assert len(sessions) == 1
    assert sessions[0]["id"] == "sess-a"
    assert sessions[0]["overall_score"] == 74


def test_get_resident_sessions_includes_score(db_path):
    from backend.persistence import create_session, get_or_create_resident, get_resident_sessions, save_evaluation

    rid = get_or_create_resident("carol@example.com")
    case = make_case()
    create_session("sess-c", "mi", "hard", case, resident_id=rid)
    save_evaluation("sess-c", make_evaluation(88), resident_id=rid)

    sessions = get_resident_sessions(rid)
    assert sessions[0]["overall_score"] == 88
    assert sessions[0]["difficulty"] == "hard"
    assert sessions[0]["status"] == "evaluated"


def test_get_resident_stats_empty(db_path):
    from backend.persistence import get_or_create_resident, get_resident_stats

    rid = get_or_create_resident("dan@example.com")
    stats = get_resident_stats(rid)
    assert stats["cases_completed"] == 0
    assert stats["avg_score"] is None
    assert stats["recommended_difficulty"] == "medium"
    assert stats["common_missed_symptoms"] == []


def test_get_resident_stats_aggregates_correctly(db_path):
    from backend.persistence import create_session, get_or_create_resident, get_resident_stats, save_evaluation

    rid = get_or_create_resident("eve@example.com")
    case = make_case()
    for i, (score, missed) in enumerate([
        (40, ["nausea", "sweating"]),
        (60, ["sweating"]),
        (80, []),
    ]):
        sid = f"stats-sess-{i}"
        create_session(sid, "mi", "medium", case, resident_id=rid)
        save_evaluation(sid, make_evaluation(score, missed), resident_id=rid)

    stats = get_resident_stats(rid)
    assert stats["cases_completed"] == 3
    assert stats["avg_score"] == round((40 + 60 + 80) / 3, 2)
    assert "sweating" in stats["common_missed_symptoms"]
    assert stats["recommended_difficulty"] == "medium"


def test_get_resident_stats_high_scores_recommend_hard(db_path):
    from backend.persistence import create_session, get_or_create_resident, get_resident_stats, save_evaluation

    rid = get_or_create_resident("frank@example.com")
    case = make_case()
    for i in range(3):
        sid = f"hard-sess-{i}"
        create_session(sid, "mi", "medium", case, resident_id=rid)
        save_evaluation(sid, make_evaluation(90), resident_id=rid)

    stats = get_resident_stats(rid)
    assert stats["recommended_difficulty"] == "hard"


def test_get_resident_stats_low_scores_recommend_easy(db_path):
    from backend.persistence import create_session, get_or_create_resident, get_resident_stats, save_evaluation

    rid = get_or_create_resident("grace@example.com")
    case = make_case()
    for i in range(3):
        sid = f"easy-sess-{i}"
        create_session(sid, "mi", "medium", case, resident_id=rid)
        save_evaluation(sid, make_evaluation(30), resident_id=rid)

    stats = get_resident_stats(rid)
    assert stats["recommended_difficulty"] == "easy"


def test_get_session_for_resident_returns_own(db_path):
    from backend.persistence import create_session, get_or_create_resident, get_session_for_resident

    rid = get_or_create_resident("henry@example.com")
    case = make_case()
    create_session("sess-h", "mi", "medium", case, resident_id=rid)

    row = get_session_for_resident("sess-h", rid)
    assert row is not None
    assert row["id"] == "sess-h"


def test_get_session_for_resident_returns_none_for_other_resident(db_path):
    from backend.persistence import create_session, get_or_create_resident, get_session_for_resident

    alice_id = get_or_create_resident("alice2@example.com")
    bob_id = get_or_create_resident("bob2@example.com")
    case = make_case()
    create_session("sess-alice", "mi", "medium", case, resident_id=alice_id)

    assert get_session_for_resident("sess-alice", bob_id) is None


def test_delete_session_for_resident(db_path):
    from backend.persistence import create_session, delete_session_for_resident, get_or_create_resident, get_resident_sessions

    rid = get_or_create_resident("ivan@example.com")
    case = make_case()
    create_session("sess-del", "mi", "medium", case, resident_id=rid)

    deleted = delete_session_for_resident("sess-del", rid)
    assert deleted is True
    assert get_resident_sessions(rid) == []


def test_delete_session_for_resident_returns_false_for_other_resident(db_path):
    from backend.persistence import create_session, delete_session_for_resident, get_or_create_resident

    alice_id = get_or_create_resident("alice3@example.com")
    bob_id = get_or_create_resident("bob3@example.com")
    case = make_case()
    create_session("sess-protected", "mi", "medium", case, resident_id=alice_id)

    deleted = delete_session_for_resident("sess-protected", bob_id)
    assert deleted is False


def test_pause_session_sets_status_and_snapshot(db_path):
    import sqlite3 as _sqlite3
    from backend.persistence import create_session, get_or_create_resident, pause_session

    rid = get_or_create_resident("julia@example.com")
    case = make_case()
    create_session("sess-pause", "mi", "medium", case, resident_id=rid)
    pause_session("sess-pause", graph_snapshot="Symptoms: chest pressure")

    conn = _sqlite3.connect(db_path)
    conn.row_factory = _sqlite3.Row
    row = conn.execute("SELECT status, graph_snapshot FROM web_sessions WHERE id='sess-pause'").fetchone()
    conn.close()
    assert row["status"] == "paused"
    assert row["graph_snapshot"] == "Symptoms: chest pressure"


# ── Dashboard endpoints ───────────────────────────────────────────────────────

def test_dashboard_sessions_endpoint_requires_auth(db_path):
    from fastapi import HTTPException
    import backend.web_app as web_app

    class UnauthRequest:
        cookies: dict = {}

    with pytest.raises(HTTPException) as exc_info:
        web_app.dashboard_sessions(UnauthRequest())
    assert exc_info.value.status_code == 401


def test_dashboard_sessions_endpoint_returns_list(db_path):
    from backend.persistence import create_session, get_or_create_resident, save_evaluation
    import backend.web_app as web_app

    rid = get_or_create_resident("kim@example.com")
    case = make_case()
    create_session("sess-k", "mi", "medium", case, resident_id=rid)
    save_evaluation("sess-k", make_evaluation(72), resident_id=rid)

    result = web_app.dashboard_sessions(make_authed_request(rid))
    assert "sessions" in result
    assert len(result["sessions"]) == 1
    assert result["sessions"][0]["overall_score"] == 72


def test_dashboard_stats_endpoint_returns_stats(db_path):
    from backend.persistence import create_session, get_or_create_resident, save_evaluation
    import backend.web_app as web_app

    rid = get_or_create_resident("leo@example.com")
    case = make_case()
    create_session("sess-l", "mi", "medium", case, resident_id=rid)
    save_evaluation("sess-l", make_evaluation(82), resident_id=rid)

    result = web_app.dashboard_stats(make_authed_request(rid))
    assert result["cases_completed"] == 1
    assert result["avg_score"] == 82.0
    assert result["recommended_difficulty"] == "hard"


def test_dashboard_session_detail_endpoint(db_path):
    from backend.persistence import create_session, get_or_create_resident, save_session
    import backend.web_app as web_app

    rid = get_or_create_resident("mia@example.com")
    case = make_case()
    session = make_web_session(case)
    session.student_questions.append("Any chest pain?")
    session.patient_responses.append("Yes.")
    create_session("sess-m", "mi", "medium", case, resident_id=rid)
    save_session("sess-m", session)

    result = web_app.dashboard_session_detail("sess-m", make_authed_request(rid))
    assert result["session_id"] == "sess-m"
    assert result["status"] == "active"
    assert len(result["qa_pairs"]) == 1
    assert result["qa_pairs"][0]["question"] == "Any chest pain?"
    assert result["case_preview"]["chief_complaint"] == "chest pressure"


def test_dashboard_session_detail_requires_ownership(db_path):
    from fastapi import HTTPException
    from backend.persistence import create_session, get_or_create_resident
    import backend.web_app as web_app

    alice_id = get_or_create_resident("alice4@example.com")
    bob_id = get_or_create_resident("bob4@example.com")
    case = make_case()
    create_session("sess-private", "mi", "medium", case, resident_id=alice_id)

    with pytest.raises(HTTPException) as exc_info:
        web_app.dashboard_session_detail("sess-private", make_authed_request(bob_id))
    assert exc_info.value.status_code == 404


def test_dashboard_delete_endpoint(db_path):
    from backend.persistence import create_session, get_or_create_resident, get_resident_sessions
    import backend.web_app as web_app

    rid = get_or_create_resident("noah@example.com")
    case = make_case()
    create_session("sess-n", "mi", "medium", case, resident_id=rid)

    result = web_app.dashboard_delete_session("sess-n", make_authed_request(rid))
    assert result["ok"] is True
    assert get_resident_sessions(rid) == []


def test_pause_endpoint(db_path):
    import sqlite3 as _sqlite3
    from backend.persistence import create_session
    import backend.web_app as web_app

    case = make_case()
    create_session("sess-pause-ep", "mi", "medium", case)
    web_app.SESSIONS["sess-pause-ep"] = make_web_session(case)

    result = web_app.pause("sess-pause-ep", web_app.PauseRequest(elapsed_seconds=0), make_anon_request())
    assert result["ok"] is True

    conn = _sqlite3.connect(db_path)
    conn.row_factory = _sqlite3.Row
    row = conn.execute("SELECT status FROM web_sessions WHERE id='sess-pause-ep'").fetchone()
    conn.close()
    assert row["status"] == "paused"


def test_resume_endpoint_restores_session(db_path):
    from backend.persistence import create_session, save_session
    import backend.web_app as web_app

    case = make_case()
    session = make_web_session(case)
    session.student_questions.append("Pain?")
    session.patient_responses.append("Yes.")
    create_session("sess-resume", "mi", "medium", case)
    save_session("sess-resume", session)

    # Clear cache to simulate restart
    web_app.SESSIONS.pop("sess-resume", None)

    result = web_app.resume("sess-resume", make_anon_request())
    assert result["session_id"] == "sess-resume"
    assert len(result["qa_pairs"]) == 1
    assert result["qa_pairs"][0]["answer"] == "Yes."
    assert "sess-resume" in web_app.SESSIONS


def test_resume_endpoint_raises_404_for_unknown(db_path):
    from fastapi import HTTPException
    import backend.web_app as web_app

    with pytest.raises(HTTPException) as exc_info:
        web_app.resume("ghost-session", make_anon_request())
    assert exc_info.value.status_code == 404


def test_resume_denied_for_non_owner(db_path):
    from fastapi import HTTPException
    from backend.persistence import create_session, get_or_create_resident, save_session
    import backend.web_app as web_app

    owner = get_or_create_resident("owner@example.com")
    intruder = get_or_create_resident("intruder@example.com")
    case = make_case()
    create_session("sess-owned", "mi", "medium", case, resident_id=owner)
    save_session("sess-owned", make_web_session(case))
    web_app.SESSIONS.pop("sess-owned", None)

    with pytest.raises(HTTPException) as exc_info:
        web_app.resume("sess-owned", make_authed_request(intruder))
    assert exc_info.value.status_code == 403
    # and the owner is still allowed in
    assert web_app.resume("sess-owned", make_authed_request(owner))["session_id"] == "sess-owned"
