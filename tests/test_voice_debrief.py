from __future__ import annotations

import time

from backend import voice_debrief
from backend.simulated_case_store import load_seed_case
from backend.simulation_engine import SimulationEngine
from backend.voice_session import VoiceSession


def _session():
    case = load_seed_case("myasthenic_crisis")
    return VoiceSession(id="s1", resident_id="r1", case=case, engine=SimulationEngine(case))


def _card(kind, title, category):
    return {"kind": kind, "id": title, "title": title, "category": category, "summary": ""}


def test_run_summary_collects_actions_and_outcome():
    s = _session()
    s.cards = [
        _card("ecg", "ECG", "ecg"),
        _card("report", "CBC", "lab"),
        _card("exam", "Neurological exam", "exam"),
        _card("consult", "Neurology consultant", "consult"),
    ]
    s.engine.run.elapsed_seconds = 125
    s.engine._enter_state("disposition", record=True)   # terminal

    summary = voice_debrief.run_summary(s)
    assert summary["ended"] is True
    assert summary["outcome_terminal"] is True
    assert summary["elapsed_label"] == "02:05"
    assert summary["investigations"] == ["ECG", "CBC"]
    assert summary["exams"] == ["Neurological exam"]
    assert summary["consults"] == ["Neurology consultant"]


def test_result_reads_surfaces_hidden_interpretation_for_ordered_tests():
    # The clinical read is withheld live; the debrief brings it back for teaching,
    # keyed on the investigation id (so only tests the resident actually ordered).
    s = _session()
    s.cards = [
        {"kind": "report", "id": "vbg", "title": "Venous blood gas", "category": "lab", "summary": ""},
    ]
    reads = voice_debrief._result_reads(s)
    assert {"test": "Venous blood gas", "read": "Hypercapnic (type-2) respiratory failure from neuromuscular fatigue. The elevated pCO2 and low-normal pH confirm ventilatory failure."} in reads
    # a test that was NOT ordered must not leak its read
    assert all(r["test"] != "CT chest (with contrast)" for r in reads)


def test_build_debrief_surfaces_teaching_and_coaching(monkeypatch):
    monkeypatch.setattr(voice_debrief, "_complete", lambda sys, usr: "Solid early workup; secure the airway sooner.")
    s = _session()
    s.cards = [_card("ecg", "ECG", "ecg")]

    debrief = voice_debrief.build_debrief(s)
    assert debrief["case_title"] == s.case.meta.title
    assert debrief["coaching_note"].startswith("Solid early")
    # the case's authored teaching material is carried through
    assert isinstance(debrief["teaching"]["teaching_points"], list)
    assert debrief["teaching"]["teaching_points"]      # the seed case has teaching points


def test_build_debrief_includes_recognition_assessment(monkeypatch):
    monkeypatch.setattr(voice_debrief, "_complete", lambda sys, usr: "note")
    s = _session()
    s.engine.diagnose("myasthenic crisis")

    debrief = voice_debrief.build_debrief(s)
    rec = next(c for c in debrief["assessment"] if c["id"] == "recognition")
    assert rec["met"] is True
    assert debrief["summary"]["working_diagnoses"] == ["myasthenic crisis"]


def test_build_debrief_catches_up_idle_time(monkeypatch):
    """Manual End after idle/patient-only time should not undercount elapsed."""
    monkeypatch.setattr(voice_debrief, "_complete", lambda sys, usr: "note")
    s = _session()
    s.last_tick = time.monotonic() - 8        # 8s of wall-time the engine hasn't ticked

    debrief = voice_debrief.build_debrief(s)
    assert debrief["summary"]["elapsed_seconds"] >= 8


def test_concluded_run_clock_is_frozen(monkeypatch):
    """Once a case has ended, building the debrief must not inflate elapsed."""
    monkeypatch.setattr(voice_debrief, "_complete", lambda sys, usr: "note")
    s = _session()
    s.engine.run.elapsed_seconds = 120
    s.engine.run.ended = True
    s.last_tick = time.monotonic() - 30       # parked 30s — must be ignored

    debrief = voice_debrief.build_debrief(s)
    assert debrief["summary"]["elapsed_seconds"] == 120


def test_idle_run_gets_honest_coaching_not_praise(monkeypatch):
    """A do-nothing run must use the honest coaching prompt, not the one that
    congratulates the resident for 'avoiding errors'."""
    seen = {}
    monkeypatch.setattr(voice_debrief, "_complete", lambda sys, usr: seen.setdefault("sys", sys) or "note")
    s = _session()  # no cards, no diagnosis, no actions

    voice_debrief.build_debrief(s)
    assert seen["sys"] == voice_debrief.COACH_SYSTEM_IDLE


def test_engaged_run_gets_the_standard_coaching(monkeypatch):
    seen = {}
    monkeypatch.setattr(voice_debrief, "_complete", lambda sys, usr: seen.setdefault("sys", sys) or "note")
    s = _session()
    s.cards = [_card("ecg", "ECG", "ecg")]

    voice_debrief.build_debrief(s)
    assert seen["sys"] == voice_debrief.COACH_SYSTEM


def test_coaching_note_fails_soft(monkeypatch):
    def boom(sys, usr):
        raise RuntimeError("llm down")
    monkeypatch.setattr(voice_debrief, "_complete", boom)
    s = _session()

    debrief = voice_debrief.build_debrief(s)
    assert debrief["coaching_note"] == ""              # empty, never an exception
    assert debrief["summary"]["outcome"]               # deterministic debrief still renders
