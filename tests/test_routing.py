"""Unit tests for the Router (backend/routing.py) — Room-of-Agents C2.

These exercise the addressee decision directly (no DB, no LLM): given an already-
classified intent and a transcript, assert the Route — who answers, whether they
are present, and why. This is the regression net that turns the playtest failures
(addressing an absent person, addressing a present bystander) into permanent guards.
"""

from __future__ import annotations

from backend.nurse_brain import NurseIntent
from backend.routing import route
from backend.simulated_case_store import load_case
from backend.simulation_engine import SimulationEngine


def _engine():
    return SimulationEngine(load_case("myasthenic_crisis"))


# ── absent vs present bystander (the two playtest bugs) ───────────────────────

def test_addressing_the_absent_husband_resolves_to_him_but_not_present():
    eng = _engine()  # husband arrives at 300s
    intent = NurseIntent(op="ask_history", target="husband_collateral")
    r = route(intent, eng, "Husband, has she been getting weaker?")
    assert r.addressee == "family"
    assert r.actor_id == "husband"
    assert r.present is False          # honest-absent, NOT routed to nurse/patient
    assert r.summon is None


def test_calling_the_absent_husband_summons_him():
    eng = _engine()
    intent = NurseIntent(op="perform_action", action_text="call the husband")
    r = route(intent, eng, "Nurse, call the husband in.")
    assert r.summon == "husband"       # the nurse is told to fetch him
    assert r.addressee == "nurse"      # she acknowledges; he enters the room
    assert r.present is True


def test_present_husband_is_addressed_directly_not_the_nurse():
    eng = _engine()
    eng.tick(305)                      # husband has arrived
    intent = NurseIntent(op="chitchat")   # interpreter under-classified
    r = route(intent, eng, "Husband, what's her background?")
    assert r.addressee == "family"     # reaches him, not laundered to the nurse
    assert r.present is True


def test_consultant_is_summoned_and_present_when_called():
    eng = _engine()
    intent = NurseIntent(op="request_consult", target="neuro")
    r = route(intent, eng, "Call neurology, please.")
    assert r.addressee == "consult"
    assert r.actor_id == "neuro"
    assert r.summon == "neuro"
    assert r.present is True


# ── staff / patient / pending (parity with the old chain) ─────────────────────

def test_unaddressed_order_routes_to_the_nurse():
    eng = _engine()
    intent = NurseIntent(op="order_medication", target="rocuronium", action_text="0.3 mg/kg")
    r = route(intent, eng, "Give a reduced dose of rocuronium.")
    assert r.addressee == "nurse"


def test_bedside_smalltalk_routes_to_the_patient():
    eng = _engine()
    intent = NurseIntent(op="chitchat")
    r = route(intent, eng, "How are you feeling right now?")
    assert r.addressee == "patient"


def test_ems_handover_routes_to_ems():
    eng = _engine()
    intent = NurseIntent(op="ask_history", target="ems_handover")
    r = route(intent, eng, "EMS, what's the story?")
    assert r.addressee == "ems"
    assert r.present is True


def test_affirmation_while_pending_stays_on_the_staff_channel():
    eng = _engine()
    eng.run.pending_action = {
        "op": "perform_action", "status": "awaiting_confirm",
        "action_text": "admit to ICU", "label": "the admission",
    }
    intent = NurseIntent(op="confirm_pending")
    r = route(intent, eng, "Yes, proceed.")
    assert r.addressee == "nurse"      # not the patient → no confirm loop (S3)


def test_forced_chip_overrides_content():
    eng = _engine()
    eng.tick(305)
    intent = NurseIntent(op="chitchat")
    r = route(intent, eng, "tell me more", forced_addressee="husband")
    assert r.actor_id == "husband"
    assert r.present is True
