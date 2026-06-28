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


def test_post_intubation_reaches_terminal_via_admission():
    # Phase E fix: post_intubation -> disposition was manual-only; an admission /
    # crisis-therapy action must now drive the case to its terminal state.
    engine = SimulationEngine(load_case("myasthenic_crisis"))
    engine.goto("post_intubation")
    engine.act(LearnerAction(kind="intervene", text="admit her to the ICU"))
    assert engine.run.current_state_id == "disposition"
    assert engine.run.ended is True


def test_ct_chest_is_gated_so_it_cannot_leak_the_thymoma_early():
    engine = SimulationEngine(load_case("myasthenic_crisis"))
    result = engine.order_investigation("ct_chest")          # at t=0 (gated to 300s)
    assert result.revealed is None                            # nothing handed over early
    assert "thymoma" not in (result.message or "").lower()    # and the message can't leak it
    # The order is recorded and scheduled to report back, not silently dropped.
    assert "ct_chest" in engine.run.ordered_investigations
    assert any(ev["inv_id"] == "ct_chest" for ev in engine.run.deferred_events)


def test_due_cues_fire_once_after_lingering():
    engine = SimulationEngine(load_case("myasthenic_crisis"))
    assert engine.due_cues() == []                            # nothing yet at t=0
    engine.goto("recognition")                                # the husband-cue state
    # The cue references the husband, so it stays gated while he's en route even after
    # the team has lingered (C1 — no "ask the husband" before he's actually here).
    engine.tick(engine.CUE_AFTER_SECONDS + 1)
    assert engine.due_cues() == []
    engine.summon("husband")                                  # now at the bedside
    fired = engine.due_cues()
    assert fired and fired[0]["actor_id"] == "nurse"          # the nurse's husband cue
    assert engine.due_cues() == []                            # each cue fires only once


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


# ── Room-of-Agents presence (RA-1, docs/room-of-agents.md C1) ─────────────────

def test_in_room_actors_are_present_from_the_start():
    engine = SimulationEngine(load_case("myasthenic_crisis"))
    present = engine.present_actor_ids()
    assert {"patient", "ems", "nurse", "rt"} <= present      # bedside team is here at t=0


def test_husband_is_absent_until_his_arrival_time():
    # The husband is authored to arrive at available_after_seconds=300.
    engine = SimulationEngine(load_case("myasthenic_crisis"))
    assert engine.is_present("husband") is False             # not here yet at t=0
    assert engine.arrival_wait("husband") == 300
    engine.tick(305)                                         # let the clock pass 300s
    assert engine.is_present("husband") is True              # arrives on the clock
    assert engine.arrival_wait("husband") is None


def test_summon_brings_a_timed_actor_in_early():
    engine = SimulationEngine(load_case("myasthenic_crisis"))
    assert engine.is_present("husband") is False
    assert engine.summon("husband") is True                  # resident actively calls him
    assert engine.is_present("husband") is True              # now in the room
    assert engine.summon("husband") is False                 # idempotent


def test_consultants_are_not_in_the_room_until_called():
    # Consultants advise by phone / arrive on request — never present by the clock.
    engine = SimulationEngine(load_case("myasthenic_crisis"))
    engine.tick(10_000)                                      # no amount of waiting summons them
    assert engine.is_present("neuro") is False
    assert engine.is_present("icu") is False
    engine.summon("neuro")
    assert engine.is_present("neuro") is True


def test_presence_summons_survive_snapshot_roundtrip():
    engine = SimulationEngine(load_case("myasthenic_crisis"))
    engine.summon("husband")
    restored = SimulationEngine(load_case("myasthenic_crisis"))
    restored.restore(engine.snapshot())
    assert restored.is_present("husband") is True
    assert restored.run.summoned_actors == ["husband"]


def test_legacy_snapshot_without_summoned_actors_restores_to_empty():
    engine = SimulationEngine(load_case("myasthenic_crisis"))
    snap = engine.snapshot()
    snap["run"].pop("summoned_actors", None)                 # snapshot predating RA-1
    restored = SimulationEngine(load_case("myasthenic_crisis"))
    restored.restore(snap)                                   # additive field → default []
    assert restored.run.summoned_actors == []


# ── Arrival announcements (C1) ────────────────────────────────────────────────

def test_arrival_announces_once_when_a_timed_actor_enters():
    engine = SimulationEngine(load_case("myasthenic_crisis"))
    assert engine.due_arrivals() == []                       # husband still en route at t=0
    engine.tick(305)                                         # clock passes his 300s arrival
    fired = engine.due_arrivals()
    assert [a["actor_id"] for a in fired] == ["husband"]     # announced the moment he arrives
    assert engine.due_arrivals() == []                       # exactly once
    assert "husband" in engine.run.announced_arrivals


def test_summon_suppresses_the_duplicate_clock_arrival_announce():
    # A summon is announced by the nurse when it happens; the later clock arrival
    # must not announce him a second time.
    engine = SimulationEngine(load_case("myasthenic_crisis"))
    engine.summon("husband")
    engine.tick(305)
    assert engine.due_arrivals() == []


def test_cue_referencing_an_absent_actor_stays_gated_until_present():
    engine = SimulationEngine(load_case("myasthenic_crisis"))
    engine.goto("recognition")
    engine.tick(80)                                          # lingered, but husband absent
    assert engine.due_cues() == []                           # the husband cue can't fire yet
    engine.tick(230)                                         # now past 300s — he's here
    assert engine.due_cues()                                 # eligible once he's present


# ── Deferred results: the room talks back (issues #4/#5) ──────────────────────

def test_ordering_a_latency_test_schedules_a_callback_not_a_dead_pending():
    engine = SimulationEngine(load_case("myasthenic_crisis"))
    step = engine.order_investigation("vital_capacity")      # RT bedside, 45s latency
    assert step.revealed is None                             # not back instantly
    assert "vital_capacity" in engine.run.ordered_investigations
    ev = next(e for e in engine.run.deferred_events if e["inv_id"] == "vital_capacity")
    assert ev["actor_id"] == "rt"                            # the RT will report it back
    assert ev["fire_at"] == 45


def test_deferred_result_fires_after_its_latency_with_the_holder_and_value():
    engine = SimulationEngine(load_case("myasthenic_crisis"))
    engine.order_investigation("vital_capacity")
    assert engine.due_deferred() == []                       # nothing yet
    engine.tick(50)                                          # past the 45s turn-around
    due = engine.due_deferred()
    assert len(due) == 1
    assert due[0]["actor_id"] == "rt"                        # the RT delivers it
    assert "VC" in (due[0]["revealed"] or "")               # the grounded number
    assert engine.due_deferred() == []                      # drained — fires once


def test_deferred_events_survive_snapshot_roundtrip():
    engine = SimulationEngine(load_case("myasthenic_crisis"))
    engine.order_investigation("nif")
    restored = SimulationEngine(load_case("myasthenic_crisis"))
    restored.restore(engine.snapshot())
    restored.tick(50)
    due = restored.due_deferred()
    assert len(due) == 1 and due[0]["actor_id"] == "rt"
