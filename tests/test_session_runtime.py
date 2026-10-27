"""Tests for the Session Runtime (autonomous clock + event bus).

All offline: the engine outputs are faked so we exercise the pure mapping, the
bus, and the lock-guarded tick without real time or network.
"""
from __future__ import annotations

import asyncio
import threading
import types
from collections import OrderedDict

import pytest

from backend import session_runtime as sr
from backend.simulation_engine import StepResult


def _session(status: str = "active", ended: bool = False, elapsed: int = 120):
    engine = types.SimpleNamespace(
        run=types.SimpleNamespace(ended=ended, elapsed_seconds=elapsed),
        current_vitals=lambda: types.SimpleNamespace(
            model_dump=lambda exclude_none=True: {"heart_rate": 120}
        ),
        due_deferred=lambda: [],          # no scheduled results in these unit ticks
        due_arrivals=lambda: [],
    )
    return types.SimpleNamespace(
        id="s1", status=status, lock=threading.Lock(), engine=engine
    )


# ── to_events (pure mapping) ──────────────────────────────────────────────────

def test_to_events_transition_is_state_change():
    r = StepResult(state_id="s2", state_name="Recognition", transitioned=True,
                   from_state="s1", message="Patient worsening", critical_error=False)
    [ev] = sr.to_events([r], [], 100)
    assert ev.kind == "state_change"
    assert ev.t == 100
    assert ev.data["state_name"] == "Recognition"
    assert ev.data["from_state"] == "s1"


def test_to_events_message_without_transition_is_condition():
    r = StepResult(state_id="s1", state_name="Baseline", transitioned=False,
                   message="SpO2 drifting down")
    [ev] = sr.to_events([r], [], 50)
    assert ev.kind == "condition" and ev.text == "SpO2 drifting down"


def test_to_events_cues_carry_speaker():
    evs = sr.to_events([], [{"speaker": "family", "text": "Is she okay?"}], 80)
    assert [e.kind for e in evs] == ["cue"]
    assert evs[0].speaker == "family"


def test_to_events_empty_when_nothing_happened():
    assert sr.to_events([], [], 10) == []


def test_to_events_attaches_vitals_to_engine_events():
    r = StepResult(state_id="s1", state_name="Baseline", transitioned=False, message="worse")
    [ev] = sr.to_events([r], [], 30, vitals={"heart_rate": 130})
    assert ev.data["vitals"] == {"heart_rate": 130}


# ── EventBus ──────────────────────────────────────────────────────────────────

def test_bus_stamps_monotonic_seq_and_ts():
    bus = sr.EventBus()
    out = bus.publish("a", [sr.SessionEvent(kind="cue", t=1, text="x")])
    assert out[0]["seq"] == 1 and out[0]["ts"]
    out2 = bus.publish("a", [sr.SessionEvent(kind="cue", t=2, text="y")])
    assert out2[0]["seq"] == 2


def test_bus_replay_returns_events_after_seq():
    bus = sr.EventBus()
    bus.publish("a", [sr.SessionEvent(kind="cue", t=1, text="x")])
    bus.publish("a", [sr.SessionEvent(kind="cue", t=2, text="y")])
    assert [e["seq"] for e in bus.replay("a", 1)] == [2]


def test_bus_overflow_drops_oldest():
    bus = sr.EventBus(maxsize=2)
    q = bus.subscribe("a")
    for i in range(5):
        bus.publish("a", [sr.SessionEvent(kind="cue", t=i, text=str(i))])
    # queue capped at 2, holding the two newest
    drained = []
    while not q.empty():
        drained.append(q.get_nowait()["text"])
    assert drained == ["3", "4"]


# ── runtime flag ──────────────────────────────────────────────────────────────

def test_runtime_enabled_flag(monkeypatch):
    monkeypatch.delenv("LIVE_RUNTIME", raising=False)
    assert sr.runtime_enabled() is False
    monkeypatch.setenv("LIVE_RUNTIME", "1")
    assert sr.runtime_enabled() is True
    monkeypatch.setenv("LIVE_RUNTIME", "off")
    assert sr.runtime_enabled() is False


# ── _tick_session (lock-guarded sync work) ───────────────────────────────────

def test_tick_skips_paused(monkeypatch):
    s = _session(status="paused")

    def _boom(_sess):
        raise AssertionError("paused session must not advance the clock")

    monkeypatch.setattr("backend.voice_session.advance_clock", _boom)
    assert sr.SessionRuntime._tick_session(s) == []


def test_tick_skips_ended(monkeypatch):
    s = _session(ended=True)
    monkeypatch.setattr("backend.voice_session.advance_clock",
                        lambda _s: (_ for _ in ()).throw(AssertionError("no tick")))
    assert sr.SessionRuntime._tick_session(s) == []


def test_tick_active_emits_and_persists(monkeypatch):
    s = _session()
    r = StepResult(state_id="s2", state_name="Recognition", transitioned=True,
                   from_state="s1", message="worse")
    monkeypatch.setattr("backend.voice_session.advance_clock", lambda _s: [r])
    monkeypatch.setattr("backend.nurse_brain._fire_cues", lambda _s: [])
    saved = {}
    monkeypatch.setattr("backend.voice_session.persist_session",
                        lambda _s, status="active": saved.update(status=status))
    evs = sr.SessionRuntime._tick_session(s)
    assert [e.kind for e in evs] == ["state_change"]
    assert saved == {"status": "active"}


def test_tick_uneventful_does_not_persist(monkeypatch):
    s = _session()
    monkeypatch.setattr("backend.voice_session.advance_clock", lambda _s: [])
    monkeypatch.setattr("backend.nurse_brain._fire_cues", lambda _s: [])
    called = {"persist": False}
    monkeypatch.setattr("backend.voice_session.persist_session",
                        lambda *a, **k: called.update(persist=True))
    assert sr.SessionRuntime._tick_session(s) == []
    assert called["persist"] is False


def test_tick_appends_case_ended_when_run_concludes(monkeypatch):
    s = _session()
    r = StepResult(state_id="term", state_name="Disposition", transitioned=True,
                   from_state="s2", message="admitted to ICU")

    def _advance(sess):
        sess.engine.run.ended = True  # terminal transition ends the run
        return [r]

    monkeypatch.setattr("backend.voice_session.advance_clock", _advance)
    monkeypatch.setattr("backend.nurse_brain._fire_cues", lambda _s: [])
    monkeypatch.setattr("backend.voice_session.persist_session", lambda *a, **k: None)
    evs = sr.SessionRuntime._tick_session(s)
    assert evs[-1].kind == "case_ended"


# ── tick_once (drives the loop body once, real async) ─────────────────────────

def test_tick_once_publishes_active_session_events(monkeypatch):
    from backend import voice_session as vs
    s = _session()
    r = StepResult(state_id="s2", state_name="Recognition", transitioned=True,
                   from_state="s1", message="worse")
    monkeypatch.setattr("backend.voice_session.advance_clock", lambda _s: [r])
    monkeypatch.setattr("backend.nurse_brain._fire_cues", lambda _s: [])
    monkeypatch.setattr("backend.voice_session.persist_session", lambda *a, **k: None)
    monkeypatch.setattr(vs, "_SESSIONS", OrderedDict({s.id: s}))

    bus = sr.EventBus()
    runtime = sr.SessionRuntime(bus)
    asyncio.run(runtime.tick_once())
    published = bus.replay(s.id, 0)
    assert [e["kind"] for e in published] == ["state_change"]


# ── WebSocket endpoint (F2 transport) ─────────────────────────────────────────

@pytest.fixture()
def live_env(tmp_path, monkeypatch):
    """A migrated, seeded DB + a real resident, with the heavy FAISS warmup in the
    app lifespan stubbed so a TestClient spins up fast."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "ws_test.db"))
    monkeypatch.setenv("APP_SECRET", "test-secret-ws")
    from backend import cxr_service, voice_session
    from backend.db import run_migrations
    from backend.persistence import get_or_create_resident
    from backend.simulated_case_repo import seed_gold_cases
    monkeypatch.setattr(cxr_service._faiss_index, "_load", lambda: None)
    run_migrations()
    seed_gold_cases()
    voice_session._SESSIONS.clear()
    return get_or_create_resident("ws@test.com")


def test_ws_streams_published_events(live_env):
    from fastapi.testclient import TestClient

    from backend import voice_session, web_app
    from backend.auth import COOKIE_NAME, make_session_cookie
    from backend.session_runtime import SessionEvent, get_bus

    session = voice_session.start_session("myasthenic_crisis", live_env)
    client = TestClient(web_app.app)
    client.cookies.set(COOKIE_NAME, make_session_cookie(live_env))
    with client, client.websocket_connect(f"/ws/voice/{session.id}") as ws:
        get_bus().publish(session.id, [
            SessionEvent(kind="cue", t=5, speaker="nurse", text="Doctor, the husband's here.")
        ])
        msg = ws.receive_json()
    assert msg["kind"] == "cue" and msg["speaker"] == "nurse"
    assert msg["text"] == "Doctor, the husband's here." and msg["seq"] == 1


def test_ws_rejects_unauthenticated(live_env):
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    from backend import voice_session, web_app

    session = voice_session.start_session("myasthenic_crisis", live_env)
    client = TestClient(web_app.app)  # no auth cookie
    with client, pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(f"/ws/voice/{session.id}"):
            pass


def test_ws_rejects_other_residents_session(live_env):
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    from backend import voice_session, web_app
    from backend.auth import COOKIE_NAME, make_session_cookie

    session = voice_session.start_session("myasthenic_crisis", live_env)
    client = TestClient(web_app.app)
    client.cookies.set(COOKIE_NAME, make_session_cookie("not-the-owner"))
    with client, pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(f"/ws/voice/{session.id}"):
            pass
