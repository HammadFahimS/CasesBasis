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


# ── roster carries each actor's case-driven arrival time (no hardcoding) ─────
def test_case_roster_exposes_arrival_time_per_actor():
    from backend.web_app import _case_roster
    case = load_seed_case("myasthenic_crisis")
    roster = {a["id"]: a for a in _case_roster(case)}
    # the husband is en route until the case-authored arrival; staff are present from t=0
    assert roster["husband"]["available_after_seconds"] == 300
    assert roster["husband"]["speaker"] == "family"
    assert roster["nurse"]["available_after_seconds"] is None
    assert roster["patient"]["available_after_seconds"] is None
    # every entry carries the field, so the front-end never has to hardcode arrival
    assert all("available_after_seconds" in a for a in roster.values())


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
    # Exercise the immediate-report path: a bedside test with no turn-around returns
    # its value in-turn, voiced by the RT. (The gold case's VC/NIF carry a latency, so
    # there they instead report back via the deferred callback — see test_live_sim.)
    inv = next(i for i in s.engine.case.investigations if i.id == "vital_capacity")
    monkeypatch.setattr(inv, "result_latency_seconds", None)   # auto-restored (cached case)
    monkeypatch.setattr(inv, "available_after_seconds", None)
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


# ── cue engine: the room prompts a lingering team, in the cue actor's voice ───
def test_lingering_fires_a_cue_voiced_by_its_actor(monkeypatch):
    monkeypatch.setattr(nurse_brain, "_complete", lambda s, u: "Heart rate's 106, still tachypnoeic.")
    monkeypatch.setattr(actor_voice, "_complete", lambda s, u: "Doctor, should we get the husband for collateral?")
    s = _session()
    s.engine.summon("husband")                      # the husband-cue is presence-gated (C1)
    s.engine.tick(s.engine.CUE_AFTER_SECONDS + 1)   # linger in baseline past the cue threshold
    turn = _resolve_turn(s, NurseIntent(op="read_vitals"), "what's her heart rate?", [])
    assert turn.cues and turn.cues[0]["speaker"] == "nurse"   # baseline cue is the nurse's
    assert "husband" in turn.cues[0]["text"].lower()


# ── the working-diagnosis chip rides the turn payload ────────────────────────
def test_turn_carries_latest_working_diagnosis(monkeypatch):
    monkeypatch.setattr(nurse_brain, "_complete", lambda s, u: "Noted, working diagnosis recorded.")
    s = _session()
    turn = _resolve_turn(s, NurseIntent(op="diagnose", action_text="myasthenic crisis"), "I think it's myasthenic crisis", [])
    assert turn.working_dx == "myasthenic crisis"
