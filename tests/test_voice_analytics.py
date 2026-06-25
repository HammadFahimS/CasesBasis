from __future__ import annotations

import pytest

from backend import voice_persistence, voice_session
from backend.sim_scoring import Criterion, overall_score


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """A migrated DB with the gold cases seeded and one real resident."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "voice_analytics.db"))
    monkeypatch.setenv("APP_SECRET", "test-secret-voice-analytics")
    from backend.db import run_migrations
    from backend.persistence import get_or_create_resident
    from backend.simulated_case_repo import seed_gold_cases
    run_migrations()
    seed_gold_cases()
    voice_session._SESSIONS.clear()
    return get_or_create_resident("analytics@test.com")


# ── overall_score ────────────────────────────────────────────────────────────
def test_overall_score_is_percent_met():
    crits = [Criterion("a", "A", True), Criterion("b", "B", False), Criterion("c", "C", True)]
    assert overall_score(crits) == 67
    assert overall_score([]) == 0
    assert overall_score([Criterion("a", "A", True)]) == 100


# ── persistence round-trip ───────────────────────────────────────────────────
def test_record_and_read_voice_evaluation(env):
    session = voice_session.start_session("myasthenic_crisis", env)
    assessment = [
        {"id": "recognition", "label": "Recognised the diagnosis", "met": True, "detail": "Named it."},
        {"id": "safety", "label": "Avoided critical errors", "met": False, "detail": "1 error."},
    ]
    voice_persistence.record_evaluation(session.id, 50, assessment, 425)

    rows = voice_persistence.get_resident_evaluations(env)
    assert len(rows) == 1
    row = rows[0]
    assert row["overall_score"] == 50
    assert row["status"] == "evaluated"           # shaped like a dashboard row
    assert row["source"] == "voice"
    assert row["duration_seconds"] == 425
    assert "Myasthenia" in row["diagnosis"]        # case title resolved from case_id

    detail = voice_persistence.get_evaluation(session.id, env)
    assert detail["overall_score"] == 50
    assert len(detail["assessment"]) == 2


def test_unscored_runs_are_not_listed_as_evaluations(env):
    voice_session.start_session("myasthenic_crisis", env)  # active, never concluded
    assert voice_persistence.get_resident_evaluations(env) == []


def test_evaluation_is_owner_scoped(env):
    from backend.persistence import get_or_create_resident
    session = voice_session.start_session("myasthenic_crisis", env)
    voice_persistence.record_evaluation(session.id, 80, [], 100)
    intruder = get_or_create_resident("intruder@test.com")
    assert voice_persistence.get_evaluation(session.id, intruder) is None


# ── dashboard merge ──────────────────────────────────────────────────────────
def test_dashboard_sessions_merges_voice_runs(env, monkeypatch):
    from backend.routers.dashboard import dashboard_sessions, dashboard_stats

    session = voice_session.start_session("myasthenic_crisis", env)
    voice_persistence.record_evaluation(session.id, 75, [
        {"id": "recognition", "label": "Recognised the diagnosis", "met": True, "detail": ""},
    ], 300)

    class _Req:  # dashboard endpoints only need the resident cookie resolver
        pass
    # Patch require_resident where the handler resolves it (its own module).
    monkeypatch.setattr("backend.routers.dashboard.require_resident", lambda req: env)

    sessions = dashboard_sessions(_Req())["sessions"]
    voice_rows = [s for s in sessions if s.get("source") == "voice"]
    assert len(voice_rows) == 1 and voice_rows[0]["overall_score"] == 75

    stats = dashboard_stats(_Req())
    assert stats["cases_completed"] >= 1
    assert stats["avg_score"] == 75
