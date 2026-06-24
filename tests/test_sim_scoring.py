from __future__ import annotations

from backend.sim_diagnostics import build_order_card
from backend.sim_scoring import score_run
from backend.simulated_case_store import load_seed_case
from backend.simulation_engine import SimulationEngine
from backend.voice_session import VoiceSession


def _session():
    case = load_seed_case("myasthenic_crisis")
    return VoiceSession(id="s1", resident_id="r1", case=case, engine=SimulationEngine(case))


def _by_id(criteria, cid):
    return next(c for c in criteria if c.id == cid)


def _maybe(criteria, cid):
    return next((c for c in criteria if c.id == cid), None)


def _order(session, drug, dose, phrase):
    session.cards.append(build_order_card(drug, dose, phrase, status="given", card_id=f"med-{drug}"))


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
    # recognition + safety first, then the case's medication rubric (none given,
    # so only the 'avoid' and 'preferred' items appear — dose_adjust is skipped)
    assert ids[:2] == ["recognition", "safety"]
    assert "med-avoid_pyridostigmine" in ids
    assert all(set(c.to_dict()) == {"id", "label", "met", "detail"} for c in criteria)


# ── medication rubric (Phase B′) ─────────────────────────────────────────────
def test_reduced_rocuronium_scores_the_dosing_nuance():
    s = _session()
    _order(s, "rocuronium", "0.6 mg/kg", "give a reduced dose of rocuronium 0.6 mg/kg")
    criteria = score_run(s)
    assert _by_id(criteria, "med-non_depolariser").met is True       # right agent
    assert _by_id(criteria, "med-rocuronium_reduced").met is True    # right direction
    assert _by_id(criteria, "med-avoid_pyridostigmine").met is True  # correctly avoided
    # succinylcholine wasn't given → its dose_adjust criterion is not shown
    assert _maybe(criteria, "med-succinylcholine_increased") is None


def test_standard_rocuronium_misses_the_dose_adjustment():
    s = _session()
    _order(s, "rocuronium", "1.2 mg/kg", "rocuronium 1.2 mg/kg")
    crit = _by_id(score_run(s), "med-rocuronium_reduced")
    assert crit.met is False
    assert "smaller dose" in crit.detail.lower()


def test_pyridostigmine_fails_the_avoid_criterion():
    s = _session()
    _order(s, "pyridostigmine", "", "give pyridostigmine")
    assert _by_id(score_run(s), "med-avoid_pyridostigmine").met is False


def test_succinylcholine_without_increase_misses_but_non_depolariser_also_misses():
    s = _session()
    _order(s, "succinylcholine", "1.5 mg/kg", "standard succinylcholine 1.5 mg/kg")
    criteria = score_run(s)
    assert _by_id(criteria, "med-succinylcholine_increased").met is False
    assert _by_id(criteria, "med-non_depolariser").met is False
