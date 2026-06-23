from __future__ import annotations

from backend.sim_scoring import score_run
from backend.simulated_case_store import load_seed_case
from backend.simulation_engine import SimulationEngine
from backend.voice_session import VoiceSession


def _session():
    case = load_seed_case("myasthenic_crisis")
    return VoiceSession(id="s1", resident_id="r1", case=case, engine=SimulationEngine(case))


def _by_id(criteria, cid):
    return next(c for c in criteria if c.id == cid)


def test_recognition_met_when_diagnosis_names_the_condition():
    s = _session()
    s.engine.diagnose("I think this is a myasthenic crisis")
    rec = _by_id(score_run(s), "recognition")
    assert rec.met is True
    assert "Myasthenic Crisis" in rec.detail or "myasthenic" in rec.detail.lower()


def test_recognition_unmet_when_no_diagnosis_stated():
    s = _session()
    rec = _by_id(score_run(s), "recognition")
    assert rec.met is False
    assert "no working diagnosis" in rec.detail.lower()


def test_recognition_unmet_on_wrong_diagnosis():
    s = _session()
    s.engine.diagnose("looks like a pulmonary embolism")
    rec = _by_id(score_run(s), "recognition")
    assert rec.met is False
    assert "match" in rec.detail.lower()


def test_safety_criterion_tracks_critical_errors():
    s = _session()
    assert _by_id(score_run(s), "safety").met is True
    s.engine.run.critical_errors.append("intubation_prep->pea_arrest: standard sux")
    assert _by_id(score_run(s), "safety").met is False


def test_score_run_is_stable_and_serialisable():
    s = _session()
    criteria = score_run(s)
    ids = [c.id for c in criteria]
    assert ids == ["recognition", "safety"]            # stable order for the UI
    assert all(set(c.to_dict()) == {"id", "label", "met", "detail"} for c in criteria)
