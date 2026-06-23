from __future__ import annotations

import pytest

from backend.simulated_case_store import CaseNotFoundError, list_cases, load_case
from backend.simulation_engine import LearnerAction, SimulationEngine
from schemas.simulated_case import SimulatedCase


def test_gold_case_loads_and_validates():
    case = load_case("myasthenic_crisis")
    assert isinstance(case, SimulatedCase)
    assert case.initial_state == "baseline"
    # Every transition points at a real state (enforced by the model validator).
    state_ids = {s.id for s in case.states}
    for state in case.states:
        for tr in state.transitions:
            assert tr.to_state in state_ids


def test_catalogue_lists_gold_case():
    ids = {c["id"] for c in list_cases()}
    assert "myasthenic_crisis" in ids


def test_missing_case_raises():
    with pytest.raises(CaseNotFoundError):
        load_case("does_not_exist")


def test_happy_path_runs_to_disposition():
    engine = SimulationEngine(load_case("myasthenic_crisis"))
    assert engine.state.id == "baseline"

    # Focused assessment + collateral history moves us to recognition.
    step = engine.act(LearnerAction(kind="exam", text="focused neuro exam"))
    assert step.transitioned and engine.state.id == "recognition"

    # Decide to intubate -> intubation prep.
    step = engine.act(LearnerAction(kind="intervene", text="prepare for intubation"))
    assert engine.state.id == "intubation_prep"

    # Correct paralytic dosing -> post-intubation.
    step = engine.act(LearnerAction(kind="medicate", text="rocuronium reduced dose"))
    assert engine.state.id == "post_intubation"
    assert not step.critical_error

    # Disposition is a manual edge in the source; facilitator/engine advances it.
    engine.goto("disposition", reason="accepted to ICU")
    assert engine.state.is_terminal and engine.run.ended


def test_critical_error_paralytic_causes_arrest_then_rosc():
    engine = SimulationEngine(load_case("myasthenic_crisis"))
    engine.act(LearnerAction(kind="exam", text="neuro exam and collateral history"))
    engine.act(LearnerAction(kind="intervene", text="intubate the patient"))
    assert engine.state.id == "intubation_prep"

    # Standard-dose succinylcholine is the critical error -> PEA arrest.
    step = engine.act(LearnerAction(kind="medicate", text="standard dose succinylcholine"))
    assert step.critical_error
    assert engine.state.id == "pea_arrest"
    assert engine.run.critical_errors

    # After ~2 minutes of resuscitation the patient achieves ROSC (time transition).
    results = engine.tick(120)
    assert any(r.transitioned for r in results)
    assert engine.state.id == "post_intubation"


def test_gated_history_and_investigation_timing():
    engine = SimulationEngine(load_case("myasthenic_crisis"))

    # Husband collateral isn't available until 5 minutes in.
    early = engine.ask("husband_collateral")
    assert early.revealed is None and "Not available" in early.message

    engine.tick(300)
    later = engine.ask("husband_collateral")
    assert later.revealed is not None

    # VBG result is pending before 2 minutes, available after.
    eng2 = SimulationEngine(load_case("myasthenic_crisis"))
    pending = eng2.order_investigation("vbg")
    assert pending.revealed is None
    eng2.tick(120)
    ready = eng2.order_investigation("vbg")
    assert ready.revealed is not None


def test_diagnose_records_working_diagnosis():
    engine = SimulationEngine(load_case("myasthenic_crisis"))
    step = engine.diagnose("myasthenic crisis")
    assert engine.run.working_diagnoses == ["myasthenic crisis"]
    assert "myasthenic crisis" in (step.revealed or "")
    # de-duplicates, and routes the same way via act(kind="diagnose")
    engine.act(LearnerAction(kind="diagnose", text="myasthenic crisis"))
    engine.diagnose("thymoma")
    assert engine.run.working_diagnoses == ["myasthenic crisis", "thymoma"]


def test_working_diagnoses_survive_snapshot_roundtrip():
    engine = SimulationEngine(load_case("myasthenic_crisis"))
    engine.diagnose("myasthenic crisis")
    restored = SimulationEngine(load_case("myasthenic_crisis"))
    restored.restore(engine.snapshot())
    assert restored.run.working_diagnoses == ["myasthenic crisis"]


def test_snapshot_roundtrip():
    engine = SimulationEngine(load_case("myasthenic_crisis"))
    engine.act(LearnerAction(kind="exam", text="neuro exam"))
    snap = engine.snapshot()

    restored = SimulationEngine(load_case("myasthenic_crisis"))
    restored.restore(snap)
    assert restored.state.id == engine.state.id
    assert restored.run.elapsed_seconds == engine.run.elapsed_seconds


def test_restore_tolerates_snapshot_drift():
    """A parked snapshot from a different RunState shape must not crash resume."""
    engine = SimulationEngine(load_case("myasthenic_crisis"))
    engine.run.elapsed_seconds = 90
    snap = engine.snapshot()
    snap["run"]["a_field_we_later_removed"] = "stale"   # unknown key → ignored
    snap["run"].pop("critical_errors", None)             # missing key → default

    restored = SimulationEngine(load_case("myasthenic_crisis"))
    restored.restore(snap)                               # no TypeError
    assert restored.run.elapsed_seconds == 90
    assert restored.run.critical_errors == []
