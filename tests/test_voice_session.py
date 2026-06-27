from __future__ import annotations

import time

import pytest

from backend.simulated_case_repo import seed_gold_cases
from backend.voice_session import (
    AccessDenied,
    CaseNotFound,
    advance_clock,
    get_session,
    start_session,
)


@pytest.fixture()
def db_path(tmp_path, monkeypatch):
    path = str(tmp_path / "voice_session_test.db")
    monkeypatch.setenv("DB_PATH", path)
    monkeypatch.setenv("APP_SECRET", "test-secret-voice-session")
    from backend.db import run_migrations
    run_migrations()
    seed_gold_cases()
    yield path


def test_start_session_builds_engine_at_initial_state(db_path):
    session = start_session("myasthenic_crisis", "resident-1")
    assert session.resident_id == "resident-1"
    assert session.engine.run.current_state_id == session.case.initial_state
    assert session.id


def test_get_session_returns_same_for_owner(db_path):
    session = start_session("myasthenic_crisis", "resident-1")
    assert get_session(session.id, "resident-1") is session


def test_get_session_rejects_other_resident(db_path):
    session = start_session("myasthenic_crisis", "resident-1")
    with pytest.raises(AccessDenied):
        get_session(session.id, "resident-2")


def test_get_session_unknown_id(db_path):
    with pytest.raises(CaseNotFound):
        get_session("nope", "resident-1")


def test_start_unknown_case(db_path):
    with pytest.raises(CaseNotFound):
        start_session("does_not_exist", "resident-1")


def test_advance_clock_moves_engine_forward(db_path):
    session = start_session("myasthenic_crisis", "resident-1")
    session.last_tick = time.monotonic() - 5
    advance_clock(session)
    assert session.engine.run.elapsed_seconds >= 5
