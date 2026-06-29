from __future__ import annotations

from backend import consult_voice
from backend.simulated_case_store import load_seed_case
from backend.simulation_engine import SimulationEngine
from backend.voice_session import VoiceSession


def _session():
    case = load_seed_case("myasthenic_crisis")
    return VoiceSession(id="s1", resident_id="r1", case=case, engine=SimulationEngine(case))


def test_consultants_excludes_patient_nurse_and_family():
    case = load_seed_case("myasthenic_crisis")
    ids = {a.id for a in consult_voice.consultants(case)}
    # the specialists are pageable; the patient/nurse/EMS/family are not — and neither
    # is the RT, who is a bedside team member holding orderable tests (VC/NIF), not a
    # phone consult. Treating "RT, get me a VC" as a consult meant it never reached
    # the engine and the numbers never came back.
    assert {"neuro", "icu"} <= ids
    assert {"patient", "nurse", "ems", "husband", "rt"}.isdisjoint(ids)


def test_find_consultant_by_id_and_role_text():
    case = load_seed_case("myasthenic_crisis")
    assert consult_voice.find_consultant(case, "neuro").id == "neuro"
    assert consult_voice.find_consultant(case, "neurology").id == "neuro"  # role overlap
    assert consult_voice.find_consultant(case, "icu").id == "icu"
    # an unknown target still resolves to *some* consultant rather than None
    assert consult_voice.find_consultant(case, "nephrology") is not None


def test_consult_reply_role_plays_and_builds_card(monkeypatch):
    captured = {}

    def fake_complete(system_prompt, user_prompt):
        captured["system"] = system_prompt
        captured["user"] = user_prompt
        # the model may fake gravitas with ellipses — they must be sanitised out
        return "This is neurology...I'd intubate early and start IVIG."

    monkeypatch.setattr(consult_voice, "_complete", fake_complete)
    session = _session()
    reply, card = consult_voice.consult_reply(session, "neuro", "call neurology please")

    # grounded: the consultant is told the presenting complaint, not asked to invent
    assert session.case.presentation.chief_complaint in captured["user"]
    assert "call neurology please" in captured["user"]
    # sanitised for speech (no pause-faking ellipses survive)
    assert reply == "This is neurology I'd intubate early and start IVIG."
    # the card routes to the Consults pane and carries the consultant + advice
    assert card["kind"] == "consult"
    assert card["category"] == "consult"
    assert card["id"] == "neuro"
    assert card["title"] == "Neurology consultant"
    assert card["summary"] == reply
    # consult_reply leaves history to the caller (run_turn records it once)
    assert session.history == []


def test_consult_reply_fails_soft(monkeypatch):
    def boom(s, u):
        raise RuntimeError("llm down")

    monkeypatch.setattr(consult_voice, "_complete", boom)
    session = _session()
    reply, card = consult_voice.consult_reply(session, "icu", "get ICU on the phone")

    assert reply  # a spoken fallback, never an exception
    assert card and card["kind"] == "consult"
    assert card["title"] == "ICU consultant"
