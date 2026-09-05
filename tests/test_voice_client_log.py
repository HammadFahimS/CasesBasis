"""Browser-side telemetry for a voice session (``/api/voice/client-log``).

The rooms post page errors and audio outcomes so a "reply shown but never heard"
run can be diagnosed from the server (2026-09-05 prod runs had no such signal).
Ownership is enforced, and both the post and each event are bounded.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from backend import voice_persistence, voice_session


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "client_log.db"))
    monkeypatch.setenv("APP_SECRET", "test-secret-client-log")
    from backend.db import run_migrations
    from backend.persistence import get_or_create_resident
    from backend.simulated_case_repo import seed_gold_cases
    run_migrations()
    seed_gold_cases()
    voice_session._SESSIONS.clear()
    return get_or_create_resident("clientlog@test.com"), get_or_create_resident("other@test.com")


class Req:
    def __init__(self, resident_id):
        from backend.auth import COOKIE_NAME, make_session_cookie
        self.cookies = {COOKIE_NAME: make_session_cookie(resident_id)}


def _session(resident_id):
    s = voice_session.start_session("myasthenic_crisis", resident_id)
    voice_session.persist_session(s)
    return s.id


def test_events_round_trip_for_the_owner(env):
    from backend.routers.voice import api_voice_client_log, api_voice_client_log_read
    me, _ = env
    sid = _session(me)
    out = api_voice_client_log(
        {"session_id": sid, "events": [
            {"kind": "tts.watchdog", "detail": "nurse: 'Four liters…' 8100ms", "t": "00:01:10"},
            {"kind": "ui.error", "detail": {"where": "engine", "message": "boom"}},
            {"kind": "", "detail": "dropped: no kind"},
            "not an event",
        ]},
        Req(me),
    )
    assert out == {"stored": 2}
    read = api_voice_client_log_read(Req(me), session_id=sid)
    kinds = [e["kind"] for e in read["events"]]
    assert kinds == ["tts.watchdog", "ui.error"]
    assert read["events"][0]["t"] == "00:01:10"
    assert '"where": "engine"' in read["events"][1]["detail"]   # dict details are serialised
    assert all(e["created_at"] for e in read["events"])


def test_events_still_accepted_after_the_session_ended(env):
    from backend.routers.voice import api_voice_client_log
    me, _ = env
    sid = _session(me)
    voice_session.end_session(sid)
    out = api_voice_client_log({"session_id": sid, "events": [{"kind": "js.error", "detail": "x"}]}, Req(me))
    assert out == {"stored": 1}


def test_other_resident_cannot_write_or_read(env):
    from backend.routers.voice import api_voice_client_log, api_voice_client_log_read
    me, other = env
    sid = _session(me)
    with pytest.raises(HTTPException) as exc:
        api_voice_client_log({"session_id": sid, "events": [{"kind": "x"}]}, Req(other))
    assert exc.value.status_code == 404
    with pytest.raises(HTTPException) as exc:
        api_voice_client_log_read(Req(other), session_id=sid)
    assert exc.value.status_code == 404
    assert voice_persistence.client_events(sid, me) == []


def test_post_is_bounded(env):
    from backend.routers.voice import api_voice_client_log
    me, _ = env
    sid = _session(me)
    huge = "x" * 10_000
    events = [{"kind": "k" * 200, "detail": huge}] * 80
    out = api_voice_client_log({"session_id": sid, "events": events}, Req(me))
    assert out == {"stored": voice_persistence.MAX_CLIENT_EVENTS_PER_POST}
    stored = voice_persistence.client_events(sid, me, limit=1000)
    assert len(stored) == voice_persistence.MAX_CLIENT_EVENTS_PER_POST
    assert len(stored[0]["detail"]) == voice_persistence.MAX_CLIENT_DETAIL_CHARS
    assert len(stored[0]["kind"]) == voice_persistence.MAX_CLIENT_KIND_CHARS


def test_bad_payloads_are_rejected(env):
    from backend.routers.voice import api_voice_client_log
    me, _ = env
    sid = _session(me)
    with pytest.raises(HTTPException) as exc:
        api_voice_client_log({"events": []}, Req(me))
    assert exc.value.status_code == 400
    with pytest.raises(HTTPException) as exc:
        api_voice_client_log({"session_id": sid, "events": "nope"}, Req(me))
    assert exc.value.status_code == 400
