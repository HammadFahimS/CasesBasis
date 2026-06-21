from __future__ import annotations

import pytest

from backend import nurse_brain
from backend.nurse_brain import (
    NurseIntent,
    dispatch,
    run_turn,
    _parse_intent,
)
from backend.simulated_case_store import load_seed_case
from backend.simulation_engine import SimulationEngine


def _engine():
    return SimulationEngine(load_seed_case("myasthenic_crisis"))


# ── pure dispatch (no LLM, no DB) ────────────────────────────────────────────
def test_dispatch_order_investigation_returns_real_result():
    engine = _engine()
    result = dispatch(NurseIntent(op="order_investigation", target="ecg"), engine, "get an ECG")
    assert result is not None
    assert "Sinus tachycardia" in (result.revealed or "")


def test_dispatch_gated_investigation_reports_pending():
    # The venous gas is gated 120s; ordering it at t=0 must not leak a result.
    engine = _engine()
    result = dispatch(NurseIntent(op="order_investigation", target="vbg"), engine, "get a gas")
    assert result is not None
    assert "pH" not in (result.revealed or "")
    assert "pending" in result.message.lower()


def test_dispatch_ask_history_reveals_real_beat():
    engine = _engine()
    result = dispatch(NurseIntent(op="ask_history", target="ems_handover"), engine, "ems story")
    assert result is not None
    assert (result.revealed or "").startswith("Called 20")


def test_dispatch_perform_action_trips_transition():
    engine = _engine()
    assert engine.run.current_state_id == "baseline"
    result = dispatch(
        NurseIntent(op="perform_action", action_text="let's get a blood gas"),
        engine, "let's get a blood gas",
    )
    assert result is not None and result.transitioned
    assert engine.run.current_state_id == "recognition"


def test_dispatch_read_vitals_does_not_mutate():
    engine = _engine()
    result = dispatch(NurseIntent(op="read_vitals"), engine, "what's her heart rate")
    assert result is None
    assert engine.run.current_state_id == "baseline"


def test_dispatch_perform_exam_reveals_and_can_transition():
    engine = _engine()
    result = dispatch(NurseIntent(op="perform_exam", target="Neurological"), engine, "neuro exam")
    assert result is not None and result.revealed
    # A neuro exam in baseline is an expected action that advances the case.
    assert engine.run.current_state_id == "recognition"


# ── LLM intent matcher (for soft action transitions) ─────────────────────────
def test_make_intent_matcher_yes(monkeypatch):
    monkeypatch.setattr(nurse_brain, "_complete", lambda s, u: "Yes.")
    matcher = nurse_brain.make_intent_matcher()
    assert matcher("secure the airway now", "the resident opts to intubate") is True


def test_make_intent_matcher_no_and_failclosed(monkeypatch):
    monkeypatch.setattr(nurse_brain, "_complete", lambda s, u: "no")
    assert nurse_brain.make_intent_matcher()("give fluids", "intubate") is False

    def boom(s, u):
        raise RuntimeError("llm down")
    monkeypatch.setattr(nurse_brain, "_complete", boom)
    assert nurse_brain.make_intent_matcher()("anything", "trigger") is False


# ── defensive JSON parsing ───────────────────────────────────────────────────
def test_parse_intent_extracts_json_from_noise():
    intent = _parse_intent('sure: {"op": "order_investigation", "target": "ecg", "action_text": ""}')
    assert intent.op == "order_investigation"
    assert intent.target == "ecg"


def test_parse_intent_falls_back_to_chitchat():
    assert _parse_intent("not json at all").op == "chitchat"
    assert _parse_intent('{"op": "teleport"}').op == "chitchat"


# ── run_turn with a stubbed LLM ──────────────────────────────────────────────
@pytest.fixture()
def db_path(tmp_path, monkeypatch):
    path = str(tmp_path / "nurse_brain_test.db")
    monkeypatch.setenv("DB_PATH", path)
    monkeypatch.setenv("APP_SECRET", "test-secret-nurse-brain")
    from backend.db import run_migrations
    from backend.simulated_case_repo import seed_gold_cases
    run_migrations()
    seed_gold_cases()
    yield path


def test_run_turn_grounds_reply_in_engine(db_path, monkeypatch):
    from backend.voice_session import start_session

    def fake_complete(system_prompt, user_prompt):
        if "routing brain" in system_prompt:
            return '{"op": "order_investigation", "target": "ecg", "action_text": ""}'
        # speak step: the real ECG result must have reached this prompt
        assert "Sinus tachycardia" in user_prompt
        return "ECG shows sinus tachycardia around 105, no ischaemic changes."

    monkeypatch.setattr(nurse_brain, "_complete", fake_complete)
    # Stub the image generator so the turn stays offline (real ECG = an LLM call).
    monkeypatch.setattr(
        nurse_brain, "build_investigation_card",
        lambda inv, vitals, patient, authored: {
            "kind": "ecg", "id": inv.id, "title": inv.name,
            "summary": authored, "visual_payload": {"type": "svg", "svg": "<svg/>"},
        },
    )

    session = start_session("myasthenic_crisis", "resident-1")
    turn = run_turn(session, "pull up the ECG")

    assert turn.intent.op == "order_investigation"
    assert "Sinus tachycardia" in (turn.revealed or "")
    assert "tachycardia" in turn.reply
    assert turn.vitals.get("heart_rate")
    assert turn.diagnostic and turn.diagnostic["kind"] == "ecg"   # drawer card produced
    # the exchange is recorded for context on the next turn
    assert session.history[-2:] == [
        {"role": "resident", "text": "pull up the ECG"},
        {"role": "nurse", "text": turn.reply},
    ]


# ── opening narration (the nurse's bedside handoff) ──────────────────────────
def test_opening_narration_phrases_and_records(db_path, monkeypatch):
    from backend.voice_session import start_session

    captured = {}

    def fake_complete(system_prompt, user_prompt):
        captured["system"] = system_prompt
        captured["user"] = user_prompt
        return "This is E.K., here with worsening shortness of breath. Begin when ready."

    monkeypatch.setattr(nurse_brain, "_complete", fake_complete)
    session = start_session("myasthenic_crisis", "resident-1")
    intro = nurse_brain.opening_narration(session)

    assert "bedside nurse" in captured["system"]
    # the chief complaint is handed to the phrasing step
    assert session.case.presentation.chief_complaint in captured["user"]
    assert intro.startswith("This is E.K.")
    # recorded on the nurse channel so the next turn has context
    assert session.history[-1] == {"role": "nurse", "text": intro}
    assert session.patient_history == []  # patient channel untouched


def test_opening_narration_falls_back_when_llm_down(db_path, monkeypatch):
    from backend.voice_session import start_session

    def boom(s, u):
        raise RuntimeError("llm down")

    monkeypatch.setattr(nurse_brain, "_complete", boom)
    session = start_session("myasthenic_crisis", "resident-1")
    intro = nurse_brain.opening_narration(session)

    # the deterministic fallback still names the chief complaint and is recorded
    assert session.case.presentation.chief_complaint in intro
    assert session.history[-1]["text"] == intro


def test_run_turn_consult_produces_consult_card(db_path, monkeypatch):
    from backend import consult_voice
    from backend.voice_session import start_session

    def fake_route(system_prompt, user_prompt):
        # the consultant catalogue must reach the router so it can pick a target
        assert "Consultants you can call" in user_prompt
        return '{"op": "request_consult", "target": "neuro", "action_text": ""}'

    monkeypatch.setattr(nurse_brain, "_complete", fake_route)
    # the consultant role-play rides its own seam
    monkeypatch.setattr(
        consult_voice, "_complete",
        lambda s, u: "Neurology here — intubate early and start IVIG.",
    )

    session = start_session("myasthenic_crisis", "resident-1")
    turn = run_turn(session, "call neurology")

    assert turn.intent.op == "request_consult"
    assert turn.speaker == "consult"                  # voiced distinctly from the nurse
    assert turn.diagnostic and turn.diagnostic["kind"] == "consult"
    assert turn.diagnostic["category"] == "consult"   # routes to the Consults pane
    assert turn.diagnostic["title"] == "Neurology consultant"
    assert "IVIG" in turn.reply
    # the consult is recorded exactly once on the nurse channel
    assert session.history[-2:] == [
        {"role": "resident", "text": "call neurology"},
        {"role": "nurse", "text": turn.reply},
    ]


def test_run_action_dispatches_without_interpret(db_path, monkeypatch):
    """A panel click goes straight to dispatch — no interpret (routing) LLM call."""
    from backend.voice_session import start_session

    def fake_complete(system_prompt, user_prompt):
        # the router must never be reached for a click
        assert "routing brain" not in system_prompt
        return "Cardiovascular exam: regular rhythm, no murmurs."

    monkeypatch.setattr(nurse_brain, "_complete", fake_complete)
    session = start_session("myasthenic_crisis", "resident-1")
    turn = nurse_brain.run_action(session, "perform_exam", "Cardiovascular")

    assert turn.intent.op == "perform_exam"
    assert turn.diagnostic and turn.diagnostic["category"] == "exam"
    # the click is recorded as a natural resident utterance
    assert session.history[-2] == {"role": "resident", "text": "Examine the Cardiovascular system."}


def test_run_action_consult_builds_card(db_path, monkeypatch):
    from backend import consult_voice
    from backend.voice_session import start_session

    monkeypatch.setattr(nurse_brain, "_complete", lambda s, u: "routing should not run")
    monkeypatch.setattr(consult_voice, "_complete", lambda s, u: "Neurology — intubate early.")

    session = start_session("myasthenic_crisis", "resident-1")
    turn = nurse_brain.run_action(session, "request_consult", "neuro")

    assert turn.diagnostic and turn.diagnostic["kind"] == "consult"
    assert turn.diagnostic["title"] == "Neurology consultant"
    assert session.history[-2]["text"] == "Call Neurology consultant."


def test_run_action_rejects_unknown_op(db_path):
    from backend.voice_session import start_session

    session = start_session("myasthenic_crisis", "resident-1")
    with pytest.raises(ValueError):
        nurse_brain.run_action(session, "teleport", "x")


def test_session_payload_menu_offers_only_case_items(db_path):
    from backend.voice_session import start_session
    from backend.web_app import _voice_session_payload

    session = start_session("myasthenic_crisis", "resident-1")
    menu = _voice_session_payload(session)["menu"]

    assert "Cardiovascular" in menu["exam_systems"]
    cats = {i["category"] for i in menu["investigations"]}
    assert {"lab", "ecg", "imaging", "bedside"} <= cats
    # consultants are pageable; the patient/nurse are not in the call list
    cids = {c["id"] for c in menu["consultants"]}
    assert {"neuro", "icu"} <= cids
    assert "patient" not in cids and "nurse" not in cids


def test_run_turn_exam_produces_card(db_path, monkeypatch):
    from backend.voice_session import start_session

    def fake_complete(system_prompt, user_prompt):
        if "routing brain" in system_prompt:
            return '{"op": "perform_exam", "target": "Neurological", "action_text": ""}'
        return "On the neuro exam she has bilateral ptosis and proximal weakness."

    monkeypatch.setattr(nurse_brain, "_complete", fake_complete)

    session = start_session("myasthenic_crisis", "resident-1")
    turn = run_turn(session, "examine her neuro")

    assert turn.intent.op == "perform_exam"
    assert turn.diagnostic and turn.diagnostic["kind"] == "exam"
    assert turn.diagnostic["title"] == "Neurological exam"
