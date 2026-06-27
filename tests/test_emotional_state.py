"""EmotionalState (multi-agent-plan Phase 5) — live per-turn affect from the engine."""

from __future__ import annotations

from backend.emotional_state import for_speaker, severity
from backend.simulated_case_store import load_case
from backend.simulation_engine import SimulationEngine


def _engine():
    return SimulationEngine(load_case("myasthenic_crisis"))


def test_professionals_stay_steady_with_no_stability_override():
    eng = _engine()
    for kind in ("ems", "rt", "consult", "nurse"):
        st = for_speaker(kind, eng)
        assert st.descriptor == "steady"
        assert st.stability is None          # keep the configured default voice


def test_patient_escalates_to_more_affect_as_the_case_deteriorates():
    eng = _engine()
    baseline = for_speaker("patient", eng).stability
    eng.goto("intubation_prep")             # respiratory muscles failing
    failing = for_speaker("patient", eng).stability
    assert failing < baseline                # lower stability = more vocal affect
    assert "four words" in for_speaker("patient", eng).instruction


def test_patient_parity_with_the_previous_hardcoded_values():
    """The numbers must match the behaviour patient_voice had before the extraction."""
    eng = _engine()
    assert for_speaker("patient", eng).stability == 0.30       # calm baseline
    eng.goto("intubation_prep")
    assert for_speaker("patient", eng).stability == 0.15       # near-failure


def test_family_worry_tracks_patient_acuity():
    eng = _engine()
    calm = for_speaker("family", eng)
    assert calm.descriptor in ("concerned", "worried", "frightened")
    calm_stab = calm.stability
    eng.goto("intubation_prep")              # patient peri-arrest
    scared = for_speaker("family", eng)
    assert scared.descriptor == "frightened"
    assert scared.stability < calm_stab      # a frightened relative is voiced with more affect


def test_severity_rises_with_acuity():
    eng = _engine()
    base = severity(eng)
    eng.goto("intubation_prep")
    assert severity(eng) > base
