from __future__ import annotations

import pytest

from backend import nurse_brain, voice_persistence, voice_session


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """A migrated DB with the gold cases seeded and one real resident."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "voice_persist.db"))
    monkeypatch.setenv("APP_SECRET", "test-secret-voice-persist")
    from backend.db import run_migrations
    from backend.persistence import get_or_create_resident
    from backend.simulated_case_repo import seed_gold_cases
    run_migrations()
    seed_gold_cases()
    voice_session._SESSIONS.clear()
    return get_or_create_resident("persist@test.com")


def _stub_llm(monkeypatch):
    # exam dispatch is deterministic; the speak step just needs *some* text
    monkeypatch.setattr(
        nurse_brain, "_complete",
        lambda s, u: ('{"op": "order_investigation", "target": "ecg", "action_text": ""}'
                      if "routing brain" in s else "ECG shows sinus tachycardia."),
    )


def test_round_trip_rehydrates_engine_transcript_and_cards(env, monkeypatch):
    _stub_llm(monkeypatch)
    monkeypatch.setattr(
        nurse_brain, "build_investigation_card",
        lambda inv, vitals, patient, authored: {
            "kind": "ecg", "id": inv.id, "title": inv.name, "category": "ecg",
            "summary": authored, "visual_payload": {"type": "svg", "svg": "<svg/>"},
        },
    )
    session = voice_session.start_session("myasthenic_crisis", env)
    sid = session.id
    nurse_brain.run_turn(session, "get an ECG")
    voice_session.persist_session(session)

    # drop the live copy → force a DB rehydrate
    voice_session._SESSIONS.clear()
    revived = voice_session.get_session(sid, env)

    assert revived.engine.run.current_state_id == session.engine.run.current_state_id
    assert "ecg" in revived.engine.run.ordered_investigations
    assert len(revived.history) == 2                  # resident + nurse
    assert [c["title"] for c in revived.cards] == ["ECG"]


def test_live_cache_is_bounded_and_eviction_is_lossless(env, monkeypatch):
    _stub_llm(monkeypatch)
    monkeypatch.setattr(voice_session, "_MAX_LIVE_SESSIONS", 3)

    ids = [voice_session.start_session("myasthenic_crisis", env).id for _ in range(5)]
    # the cache never exceeds the cap…
    assert len(voice_session._SESSIONS) == 3
    # …and the earliest, now-evicted runs are gone from memory…
    assert ids[0] not in voice_session._SESSIONS
    # …but still fully recoverable from the DB.
    assert voice_session.get_session(ids[0], env).id == ids[0]


def test_resume_freezes_the_case_clock(env, monkeypatch):
    _stub_llm(monkeypatch)
    session = voice_session.start_session("myasthenic_crisis", env)
    session.engine.run.elapsed_seconds = 300          # 5 min into the case
    voice_session.persist_session(session)
    sid = session.id

    voice_session._SESSIONS.clear()
    revived = voice_session.get_session(sid, env)
    # parked wall-time must NOT advance the engine; elapsed is preserved exactly
    assert revived.engine.run.elapsed_seconds == 300


def test_ownership_is_enforced_on_rehydrate(env, monkeypatch):
    _stub_llm(monkeypatch)
    from backend.persistence import get_or_create_resident
    other = get_or_create_resident("intruder@test.com")
    session = voice_session.start_session("myasthenic_crisis", env)
    voice_session.persist_session(session)
    voice_session._SESSIONS.clear()
    with pytest.raises(voice_session.AccessDenied):
        voice_session.get_session(session.id, other)


def test_end_session_marks_ended_and_lists_for_resident(env, monkeypatch):
    _stub_llm(monkeypatch)
    session = voice_session.start_session("myasthenic_crisis", env)
    sid = session.id
    voice_session.end_session(sid)

    rows = voice_persistence.list_for_resident(env)
    row = next(r for r in rows if r["id"] == sid)
    assert row["status"] == "ended"
    assert row["ended_at"] is not None
    # the live copy is gone, but a rehydrate from the ended row still works
    assert sid not in voice_session._SESSIONS
    assert voice_persistence.load(sid)["status"] == "ended"


def test_pause_marks_session_paused(env, monkeypatch):
    _stub_llm(monkeypatch)
    from backend.auth import COOKIE_NAME, make_session_cookie
    from backend.web_app import api_voice_pause

    session = voice_session.start_session("myasthenic_crisis", env)

    class Req:
        cookies = {COOKIE_NAME: make_session_cookie(env)}

    assert api_voice_pause({"session_id": session.id}, Req()) == {"ok": True}
    assert voice_persistence.load(session.id)["status"] == "paused"
    # a paused run is still resumable from the home surface
    row = next(r for r in voice_persistence.list_for_resident(env) if r["id"] == session.id)
    assert row["status"] == "paused"


def test_sessions_endpoint_enriches_title_and_resumable(env, monkeypatch):
    """The home surface needs a case title and a resumable flag per run."""
    _stub_llm(monkeypatch)
    from backend.auth import COOKIE_NAME, make_session_cookie
    from backend.web_app import api_voice_sessions

    active = voice_session.start_session("myasthenic_crisis", env)
    ended = voice_session.start_session("myasthenic_crisis", env)
    voice_session.end_session(ended.id)

    class Req:
        cookies = {COOKIE_NAME: make_session_cookie(env)}

    out = {s["id"]: s for s in api_voice_sessions(Req())["sessions"]}
    assert out[active.id]["resumable"] is True
    assert out[ended.id]["resumable"] is False
    # case_id resolves to the human title for the card
    assert out[active.id]["title"] == "Myasthenia Gravis with Myasthenic Crisis"


def test_delete_is_owner_scoped_and_drops_the_run(env, monkeypatch):
    _stub_llm(monkeypatch)
    from fastapi import HTTPException

    from backend.auth import COOKIE_NAME, make_session_cookie
    from backend.persistence import get_or_create_resident
    from backend.web_app import api_voice_session_delete

    session = voice_session.start_session("myasthenic_crisis", env)
    other = get_or_create_resident("intruder@test.com")

    class Req:
        def __init__(self, rid):
            self.cookies = {COOKIE_NAME: make_session_cookie(rid)}

    # a different resident cannot delete it
    with pytest.raises(HTTPException) as exc:
        api_voice_session_delete(session.id, Req(other))
    assert exc.value.status_code == 404
    assert voice_persistence.load(session.id) is not None

    # the owner can — row gone and the live copy dropped
    assert api_voice_session_delete(session.id, Req(env)) == {"ok": True}
    assert voice_persistence.load(session.id) is None
    assert session.id not in voice_session._SESSIONS
