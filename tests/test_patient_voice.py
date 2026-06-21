from __future__ import annotations

import pytest

from backend import patient_voice
from backend.simulated_case_store import load_seed_case
from backend.simulation_engine import SimulationEngine
from backend.voice_session import VoiceSession


def _session():
    case = load_seed_case("myasthenic_crisis")
    return VoiceSession(id="s1", resident_id="r1", case=case, engine=SimulationEngine(case))


def test_persona_includes_patient_identity():
    case = load_seed_case("myasthenic_crisis")
    persona = patient_voice._persona(case)
    assert str(case.patient.age) in persona
    assert case.presentation.chief_complaint in persona


def test_patient_reply_records_history_and_uses_llm(monkeypatch):
    captured = {}

    def fake_complete(system_prompt, user_prompt):
        captured["system"] = system_prompt
        captured["user"] = user_prompt
        # the model loves to fake breathlessness with ellipses — they must not
        # survive into the spoken reply or the transcript.
        return "I...can't...catch...my...breath."

    monkeypatch.setattr(patient_voice, "_complete", fake_complete)
    session = _session()
    reply = patient_voice.patient_reply(session, "What brought you in tonight?")

    assert reply == "I can't catch my breath."
    assert "role-playing a patient" in captured["system"]
    assert "What brought you in tonight?" in captured["user"]
    # the patient channel keeps its own history, separate from the nurse channel
    assert session.patient_history[-2:] == [
        {"role": "resident", "text": "What brought you in tonight?"},
        {"role": "patient", "text": "I can't catch my breath."},
    ]
    assert session.history == []  # nurse channel untouched


def test_clean_for_speech_removes_pause_fakers():
    clean = patient_voice._clean_for_speech
    # ellipses, em-dashes and newlines that the TTS would read as hard pauses
    assert clean("I...feel...short...of...breath") == "I feel short of breath"
    assert clean("tight chest — when I breathe") == "tight chest when I breathe"
    assert clean(". . . it really hurts") == "it really hurts"
    assert clean("my chest hurts\nand I feel dizzy") == "my chest hurts and I feel dizzy"
    assert clean("please … help me") == "please help me"
    # real hyphenated words and ordinary sentences are left alone
    assert clean("I had an x-ray at my follow-up.") == "I had an x-ray at my follow-up."
    assert clean("It started yesterday, around noon.") == "It started yesterday, around noon."


def test_patient_reply_fails_soft(monkeypatch):
    def boom(s, u):
        raise RuntimeError("llm down")

    monkeypatch.setattr(patient_voice, "_complete", boom)
    session = _session()
    reply = patient_voice.patient_reply(session, "hello?")
    assert reply  # a spoken fallback, never an exception
    assert session.patient_history[-1]["role"] == "patient"
