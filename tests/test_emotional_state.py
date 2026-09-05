"""EmotionalState (multi-agent-plan Phase 5) — live per-turn affect from the engine.

The register is derived from the vitals the current state carries (SpO₂, GCS,
rhythm), never from a state id, so any case's arrest / intubation / recovery
states sound right without naming them in code."""

from __future__ import annotations

from backend.emotional_state import for_speaker, is_silent, severity
from backend.simulated_case_store import load_case
from backend.simulation_engine import SimulationEngine


def _engine(case_id="myasthenic_crisis"):
    return SimulationEngine(load_case(case_id))


def _set(eng, **vitals):
    eng.run.vitals_override.update(vitals)


def test_professionals_stay_steady_with_no_stability_override():
    eng = _engine()
    for kind in ("ems", "rt", "consult", "nurse"):
        st = for_speaker(kind, eng)
        assert st.descriptor == "steady"
        assert st.stability is None          # keep the configured default voice


def test_patient_escalates_with_falling_saturation():
    eng = _engine()
    baseline = for_speaker("patient", eng)
    assert baseline.stability == 0.30 and baseline.descriptor == "steady"
    _set(eng, oxygen_saturation=91)
    worse = for_speaker("patient", eng)
    assert worse.descriptor == "breathless" and worse.stability < baseline.stability
    _set(eng, oxygen_saturation=84)
    failing = for_speaker("patient", eng)
    assert failing.descriptor == "drowsy" and failing.stability < worse.stability
    assert "four words" in failing.instruction


def test_depressed_gcs_makes_the_patient_drowsy_then_silent():
    eng = _engine("foreign_body_airway")
    eng.goto("deteriorating")                 # gcs 11, SpO₂ 84 — perfusing but somnolent
    assert for_speaker("patient", eng).descriptor == "drowsy"
    assert not is_silent(eng)
    _set(eng, gcs="7")
    assert is_silent(eng)


def test_intubated_and_pulseless_patients_are_silent():
    eng = _engine()
    for sid in ("post_intubation", "pea_arrest", "disposition"):
        eng.goto(sid)
        assert is_silent(eng), sid
        assert "silence" in for_speaker("patient", eng).instruction
    fb = _engine("foreign_body_airway")
    for sid in ("pea_arrest", "death", "intubated_displaced", "rosc"):
        fb.goto(sid)
        assert is_silent(fb), sid


def test_recovered_patient_speaks_normally_again():
    fb = _engine("foreign_body_airway")
    fb.goto("deteriorating")
    assert for_speaker("patient", fb).descriptor == "drowsy"
    fb.goto("recovery")                        # SpO₂ 99, GCS 15, terminal
    st = for_speaker("patient", fb)
    assert st.descriptor == "steady" and "speak normally" in st.instruction


def test_family_worry_tracks_patient_acuity():
    eng = _engine()
    calm = for_speaker("family", eng)
    assert calm.descriptor == "concerned"
    eng.goto("pea_arrest")                     # peri-arrest
    scared = for_speaker("family", eng)
    assert scared.descriptor == "frightened"
    assert scared.stability < calm.stability   # a frightened relative is voiced with more affect


def test_severity_rises_with_acuity():
    eng = _engine()
    base = severity(eng)
    _set(eng, oxygen_saturation=88)
    mid = severity(eng)
    eng.goto("pea_arrest")
    assert base < mid < severity(eng) == 1.0
