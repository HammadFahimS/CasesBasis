from __future__ import annotations

from backend import actor_voice, nurse_brain
from backend.actor_voice import provider_speaker
from backend.nurse_brain import NurseIntent, _resolve_turn
from backend.simulated_case_store import load_seed_case
from backend.simulation_engine import SimulationEngine
from backend.voice_session import VoiceSession


def _session():
    case = load_seed_case("myasthenic_crisis")
    return VoiceSession(id="s1", resident_id="r1", case=case, engine=SimulationEngine(case))


# ── provider → speaker class ─────────────────────────────────────────────────
def test_provider_speaker_mapping():
    assert provider_speaker("ems") == "ems"
    assert provider_speaker("family") == "family"
    assert provider_speaker("patient") == "patient"
    assert provider_speaker("nurse") == "nurse"
    assert provider_speaker(None) == "nurse"
    assert provider_speaker("lab") == "nurse"   # unknown → nurse reads it


# ── a history beat is voiced by its provider, in their own speaker tag ───────
def test_ask_history_is_voiced_by_the_family_member(monkeypatch):
    monkeypatch.setattr(actor_voice, "_complete", lambda s, u: "She's been getting weaker for months, doctor.")
    s = _session()
    s.engine.run.elapsed_seconds = 9999   # clear the husband_collateral 300s gate
    intent = NurseIntent(op="ask_history", target="husband_collateral")
    turn = _resolve_turn(s, intent, "what happened to her?", [])
    assert turn.speaker == "family"
    assert "weaker" in turn.reply


def test_ask_history_ems_handover_is_voiced_by_ems(monkeypatch):
    monkeypatch.setattr(actor_voice, "_complete", lambda s, u: "Called twenty minutes ago, found her tripoding.")
    s = _session()
    intent = NurseIntent(op="ask_history", target="ems_handover")
    turn = _resolve_turn(s, intent, "what's the story?", [])
    assert turn.speaker == "ems"


# ── bedside measures (VC/NIF) are reported by the RT ─────────────────────────
def test_bedside_investigation_is_reported_by_rt(monkeypatch):
    monkeypatch.setattr(actor_voice, "_complete", lambda s, u: "Vital capacity is under twenty mils per kilo.")
    s = _session()
    s.engine.run.elapsed_seconds = 9999   # clear the 60s gate on vital_capacity
    intent = NurseIntent(op="order_investigation", target="vital_capacity")
    turn = _resolve_turn(s, intent, "get me a vital capacity", [])
    assert turn.speaker == "rt"
    # still produces the investigation card for the drawer
    assert turn.diagnostic and turn.diagnostic["category"] == "bedside"


def test_lab_order_is_still_voiced_by_the_nurse(monkeypatch):
    monkeypatch.setattr(nurse_brain, "_complete", lambda s, u: "CBC and lytes are back, all normal.")
    s = _session()
    intent = NurseIntent(op="order_investigation", target="cbc_lytes")
    turn = _resolve_turn(s, intent, "send a CBC", [])
    assert turn.speaker == "nurse"
