"""Headless live-sim conversation tests (Phase 1 of docs/multi-agent-plan.md).

Two layers, both run with ``pytest -k live_sim`` — no browser, no audio, no LLM:

1. **Conversation scripts** (``tests/sim_scripts/*.yaml``) — gold runs (the
   resident does well) and failing runs (poor performance handled correctly).
   Each plays through the real server pipeline and asserts structured state.
2. **Characterization guards** (S1–S3) — encode the three known defects as
   ``xfail(strict=True)`` assertions of the *correct* behaviour, so they prove
   the bug today and flip to passing (alarming on xpass) as Phases 2–3 land.
"""

from __future__ import annotations

import pytest

from tests import sim_harness


# ── DB + case seed (shared shape with test_nurse_brain.py) ────────────────────
@pytest.fixture()
def db_path(tmp_path, monkeypatch):
    path = str(tmp_path / "live_sim_test.db")
    monkeypatch.setenv("DB_PATH", path)
    monkeypatch.setenv("APP_SECRET", "test-secret-live-sim")
    from backend.db import run_migrations
    from backend.simulated_case_repo import seed_gold_cases
    run_migrations()
    seed_gold_cases()
    yield path


# ── 1 · conversation scripts ──────────────────────────────────────────────────
@pytest.mark.parametrize(
    "script_path", sim_harness.list_scripts(), ids=lambda p: p.stem
)
def test_live_sim_conversation_script(script_path, db_path, monkeypatch):
    script = sim_harness.load_script(script_path)
    run = sim_harness.run_script(script, monkeypatch)

    fails: list[str] = []
    spoken = [s for s in script.get("turns", []) if "say" in s or "action" in s]
    played = [t for t in run.turns if t.endpoint not in ("goto", "clock")]
    for step, tr in zip(spoken, played):
        if "expect" in step:
            fails += sim_harness.check_turn(tr, step["expect"])
    if "final" in script:
        fails += sim_harness.check_final(run, script["final"])

    assert not fails, "\n".join([f"{script['name']}:"] + fails)


# ── 2 · characterization guards for the three structural defects ──────────────
# S1 and S2 are FIXED in Phase 2 (server-side Director routing + real-speaker
# persistence) and now assert the correct behaviour directly. S3 remains xfail
# until Phase 3 (turn-spanning pending_action).

def test_s1_misaddressed_ems_recovers_via_content_not_the_patient(db_path, monkeypatch):
    """S1 fixed: routing is server-side and content-based. "Emma" (an ASR slip for
    "EMS") no longer falls to the patient default — the *story* content reaches the
    EMS voice through the handover beat."""
    stub = sim_harness.StubLLM()
    sim_harness.install_stubs(monkeypatch, stub)
    from backend.voice_session import start_session
    session = start_session("myasthenic_crisis", "resident-1")

    # The interpreter grounds "what's the story?" to the EMS handover beat.
    stub.intent_json = sim_harness._intent_json(
        {"op": "ask_history", "target": "ems_handover"}
    )
    turn = sim_harness.run_turn(session, "Emma, what's the story?")
    assert turn.speaker == "ems"           # NOT the patient — the S1 regression guard


def test_s1_staff_order_never_routes_to_the_patient(db_path, monkeypatch):
    """S1 fixed: an unaddressed staff order routes to the nurse by intent, never to
    the patient (the original 'call EMS → patient answers' failure)."""
    stub = sim_harness.StubLLM()
    sim_harness.install_stubs(monkeypatch, stub)
    from backend.voice_session import start_session
    session = start_session("myasthenic_crisis", "resident-1")
    session.engine.goto("intubation_prep")

    stub.intent_json = sim_harness._intent_json(
        {"op": "order_medication", "target": "rocuronium", "action_text": "0.3 mg/kg"}
    )
    turn = sim_harness.run_turn(session, "Give a reduced dose of rocuronium.")
    assert turn.speaker == "nurse"


def test_s2_bystander_reply_persists_under_its_own_speaker(db_path, monkeypatch):
    """S2 fixed: the transcript records who actually spoke, not a blanket 'nurse'."""
    stub = sim_harness.StubLLM()
    sim_harness.install_stubs(monkeypatch, stub)
    from backend.voice_session import start_session
    session = start_session("myasthenic_crisis", "resident-1")

    stub.intent_json = sim_harness._intent_json(
        {"op": "ask_history", "target": "ems_handover"}
    )
    turn = sim_harness.run_turn(session, "EMS, what's the story?")
    assert turn.speaker == "ems"
    assert session.history[-1]["role"] == "ems"     # real identity, not "nurse"


def test_present_actor_refuses_in_character_when_asked_outside_their_knowledge(db_path, monkeypatch):
    """C3: the RT is in the room but holds no history. Asked for past medical history,
    he declines in character — the nurse does NOT cover for him, and nothing in the
    engine advances."""
    stub = sim_harness.StubLLM()
    sim_harness.install_stubs(monkeypatch, stub)
    from backend.voice_session import start_session
    session = start_session("myasthenic_crisis", "resident-1")

    stub.intent_json = sim_harness._intent_json({"op": "chitchat"})
    turn = sim_harness.run_turn(session, "RT, what's her past medical history?")
    assert turn.speaker == "rt"                       # he answers for himself…
    from backend.agents import refuse
    assert turn.reply == refuse("rt")                 # …with the honest in-character decline
    assert session.history[-1]["role"] == "rt"        # not laundered as the nurse
    assert turn.transitioned is False


def test_visual_study_announces_availability_and_never_reads_the_interpretation(db_path, monkeypatch):
    """C4: ordering an ECG / chest film yields a deterministic 'it's up for you to
    read' line — the interpretation stays on the card, never voiced, and the spoken
    reply never goes through the embellishing LLM (which returns the [spoken reply]
    marker under the stub)."""
    stub = sim_harness.StubLLM()
    sim_harness.install_stubs(monkeypatch, stub)
    from backend.voice_session import start_session
    session = start_session("myasthenic_crisis", "resident-1")

    for inv_id, marker, leak in (
        ("ecg", "monitor", "sinus tach"),
        ("cxr", "screen", "consolidation"),
    ):
        stub.intent_json = sim_harness._intent_json(
            {"op": "order_investigation", "target": inv_id}
        )
        turn = sim_harness.run_turn(session, f"Get me {'an ECG' if inv_id == 'ecg' else 'a chest film'}.")
        assert turn.speaker == "nurse"
        assert marker in turn.reply.lower()              # an availability announce
        assert turn.reply != "[spoken reply]"            # not routed through the LLM
        assert leak not in turn.reply.lower()            # interpretation never voiced
        assert turn.diagnostic is not None               # but the card carries the detail


def test_lab_result_is_still_spoken(db_path, monkeypatch):
    """Labs come back verbally — a VBG is reported, not announced-only (only visual
    studies are card-only)."""
    stub = sim_harness.StubLLM()
    sim_harness.install_stubs(monkeypatch, stub)
    from backend.voice_session import start_session
    session = start_session("myasthenic_crisis", "resident-1")

    stub.intent_json = sim_harness._intent_json({"op": "order_investigation", "target": "vbg"})
    turn = sim_harness.run_turn(session, "Send a venous blood gas.")
    assert turn.reply == "[spoken reply]"                # spoken through the nurse, not announce-only


def test_s3_high_alert_med_without_dose_is_not_administered(db_path, monkeypatch):
    """S3 fixed: an un-dosed high-alert med parks (awaiting_info) instead of both
    asking-to-confirm AND silently executing — the case does not advance until the
    order is completed."""
    stub = sim_harness.StubLLM()
    sim_harness.install_stubs(monkeypatch, stub)
    from backend.voice_session import start_session
    session = start_session("myasthenic_crisis", "resident-1")
    session.engine.goto("intubation_prep")

    stub.intent_json = sim_harness._intent_json(
        {"op": "order_medication", "target": "rocuronium", "action_text": ""}
    )
    turn = sim_harness.run_turn(session, "Nurse, give rocuronium.")
    assert turn.transitioned is False
    assert session.engine.run.current_state_id == "intubation_prep"
    pending = session.engine.run.pending_action
    assert pending and pending["target"] == "rocuronium" and pending["status"] == "awaiting_info"


def test_s3_supplying_the_dose_administers_without_looping(db_path, monkeypatch):
    """The follow-up dose fills the parked order and executes it — no re-ask loop."""
    stub = sim_harness.StubLLM()
    sim_harness.install_stubs(monkeypatch, stub)
    from backend.voice_session import start_session
    session = start_session("myasthenic_crisis", "resident-1")
    session.engine.goto("intubation_prep")

    stub.intent_json = sim_harness._intent_json(
        {"op": "order_medication", "target": "rocuronium", "action_text": ""}
    )
    sim_harness.run_turn(session, "Give rocuronium.")          # parks awaiting_info
    # The resident answers the nurse's "how much?" — a bare dose, classified chitchat.
    stub.intent_json = sim_harness._intent_json({"op": "chitchat"})
    turn = sim_harness.run_turn(session, "Point six — 0.6 mg/kg.")
    assert turn.speaker == "nurse"                              # not the patient (S3 routing)
    assert session.engine.run.pending_action is None           # parked order cleared
    assert session.engine.run.current_state_id == "post_intubation"   # administered → advanced


# ── Room-of-Agents presence routing (RA-1/RA-2, docs/room-of-agents.md) ───────
# The two failures from the live playtest of session brw6ct06CEYorny-: addressing
# an absent person fell to nurse filler; addressing a present bystander was answered
# by the nurse covering for them. Both are now permanent guards.

def test_addressing_an_absent_actor_is_honest_not_nurse_filler(db_path, monkeypatch):
    """Asking the husband before he arrives gets an honest 'he isn't here yet' from
    the nurse — NOT a fabricated 'no new information' reply, and NOT routed to the
    patient. The engine does not advance and he is not silently summoned."""
    stub = sim_harness.StubLLM()
    sim_harness.install_stubs(monkeypatch, stub)
    from backend.voice_session import start_session
    session = start_session("myasthenic_crisis", "resident-1")   # t=0, husband absent

    stub.intent_json = sim_harness._intent_json(
        {"op": "ask_history", "target": "husband_collateral"}
    )
    turn = sim_harness.run_turn(session, "Husband, has she been getting weaker?")
    assert turn.speaker == "nurse"
    assert "isn't here" in turn.reply or "not at the bedside" in turn.reply
    assert "husband" not in session.engine.run.summoned_actors
    assert session.engine.is_present("husband") is False          # still not in the room


def test_calling_the_husband_brings_him_in(db_path, monkeypatch):
    """'Call the husband' summons him (nurse acknowledges); he is then in the room
    and a direct address reaches HIM, not the nurse."""
    stub = sim_harness.StubLLM()
    sim_harness.install_stubs(monkeypatch, stub)
    from backend.voice_session import start_session
    session = start_session("myasthenic_crisis", "resident-1")

    stub.intent_json = sim_harness._intent_json(
        {"op": "perform_action", "action_text": "call the husband"}
    )
    turn = sim_harness.run_turn(session, "Nurse, call the husband in.")
    assert turn.speaker == "nurse"
    assert session.engine.is_present("husband") is True           # now at the bedside

    # Now he is here — addressing him reaches the family voice, not nurse filler.
    stub.intent_json = sim_harness._intent_json(
        {"op": "ask_history", "target": "husband_collateral"}
    )
    turn = sim_harness.run_turn(session, "Husband, what's her background?")
    assert turn.speaker == "family"
    assert session.history[-1]["role"] == "family"


def test_present_bystander_answers_even_when_interpreter_says_chitchat(db_path, monkeypatch):
    """After the husband has arrived, a direct address the interpreter under-classified
    as chitchat still reaches him (positive routing), never the nurse covering (C3)."""
    stub = sim_harness.StubLLM()
    sim_harness.install_stubs(monkeypatch, stub)
    from backend.voice_session import start_session
    session = start_session("myasthenic_crisis", "resident-1")
    session.engine.tick(305)                                      # husband arrives on the clock

    stub.intent_json = sim_harness._intent_json({"op": "chitchat"})
    turn = sim_harness.run_turn(session, "Husband, tell me about her last few weeks.")
    assert turn.speaker == "family"                               # not nurse


def test_rt_is_not_a_phone_consultant_so_bedside_orders_reach_the_engine(db_path, monkeypatch):
    """Regression for the playtest bug: 'RT, get me a vital capacity' was classified
    request_consult(rt) and role-played into an empty 'I'll get it' that never reached
    the engine — VC/NIF never came back. The RT must not be a pageable consultant."""
    from backend.consult_voice import consultants
    from backend.simulated_case_store import load_case
    case = load_case("myasthenic_crisis")
    ids = {a.id for a in consultants(case)}
    assert "rt" not in ids
    assert {"neuro", "icu"} <= ids                                # real consultants still listed


def test_bedside_order_reports_back_later_with_the_rt_and_a_card(db_path, monkeypatch):
    """Issues #4/#5: ordering bedside spirometry schedules a deferred result; once its
    turn-around passes, the RT reports the number aloud and a card lands in the panel."""
    stub = sim_harness.StubLLM()
    sim_harness.install_stubs(monkeypatch, stub)
    from backend.voice_session import start_session
    session = start_session("myasthenic_crisis", "resident-1")

    stub.intent_json = sim_harness._intent_json(
        {"op": "order_investigation", "target": "vital_capacity"}
    )
    turn = sim_harness.run_turn(session, "RT, get me a vital capacity.")
    assert turn.revealed is None                                  # not back instantly
    assert turn.speaker == "rt"                                   # the RT acks the order in
    assert session.history[-1]["role"] == "rt"                    # their own voice, not the nurse
    assert any(ev["inv_id"] == "vital_capacity" for ev in session.engine.run.deferred_events)

    # While in flight, the room shows the RT working — pending lists the order under
    # the holder's speaker class so the front-end lights the right seat with a spinner.
    from backend.director import fire_deferred, pending_deferred
    pend = pending_deferred(session)
    assert len(pend) == 1
    assert pend[0]["speaker"] == "rt" and pend[0]["wait_seconds"] > 0

    assert fire_deferred(session) == []                          # nothing due yet
    assert pending_deferred(session)                             # still in flight
    session.engine.tick(50)                                       # past the 45s turn-around
    rows = fire_deferred(session)
    assert len(rows) == 1
    assert rows[0]["speaker"] == "rt"                            # the RT delivers it
    assert "VC" in rows[0]["text"]                                # the grounded value
    assert rows[0]["card"] and rows[0]["card"]["category"] == "bedside"
    assert session.cards[-1]["id"] == "vital_capacity"           # card landed in the drawer
    assert session.history[-1]["role"] == "rt"                   # spoken under the RT
    assert pending_deferred(session) == []                       # drained — RT no longer working


def test_arrival_is_announced_by_the_nurse_when_the_husband_walks_in(db_path, monkeypatch):
    """The 'her husband's just arrived' line fires when presence actually flips true,
    via the cue-firing path (so both the turn and the live WS surface it). The gold
    case's authored husband cue is that line: the nurse voices it, with no second
    generic 'just arrived' alongside."""
    stub = sim_harness.StubLLM()
    sim_harness.install_stubs(monkeypatch, stub)
    from backend import actor_voice
    directions: list[str] = []

    def capture(system: str, user: str) -> str:
        directions.append(user)
        return stub.voice_complete(system, user)

    monkeypatch.setattr(actor_voice, "_complete", capture)
    from backend.voice_session import start_session
    session = start_session("myasthenic_crisis", "resident-1")
    session.engine.tick(305)                                      # husband arrives on the clock

    from backend.director import fire_cues
    rows = fire_cues(session)
    assert len(rows) == 1 and rows[0]["speaker"] == "nurse"       # one line, from the nurse
    assert len(directions) == 1 and "arrived" in directions[0].lower()
    # A second pass never re-announces his arrival.
    assert fire_cues(session) == []


def test_s3_proceed_confirms_a_disposition_without_looping(db_path, monkeypatch):
    """A bare 'proceed' resolves a parked disposition (the ICU-admission loop)."""
    stub = sim_harness.StubLLM()
    sim_harness.install_stubs(monkeypatch, stub)
    from backend.voice_session import start_session
    session = start_session("myasthenic_crisis", "resident-1")
    session.engine.goto("post_intubation")

    stub.intent_json = sim_harness._intent_json(
        {"op": "perform_action", "action_text": "admit to ICU on IVIG and steroids"}
    )
    turn = sim_harness.run_turn(session, "Admit her to the ICU on IVIG and steroids.")
    assert turn.transitioned is False                          # parked, read back first
    assert session.engine.run.pending_action["status"] == "awaiting_confirm"

    stub.intent_json = sim_harness._intent_json({"op": "confirm_pending"})
    turn = sim_harness.run_turn(session, "Yes, proceed.")
    assert turn.speaker == "nurse"
    assert session.engine.run.pending_action is None
    assert session.engine.run.current_state_id == "disposition"


# ── multi-beat patient channel ────────────────────────────────────────────────
def test_patient_channel_honours_the_interpreters_beat_not_just_the_next_one(db_path, monkeypatch):
    """A case with several patient-held beats: asking about medications reveals the
    medications beat (the interpreter mapped it), not whichever beat is next; a
    label-word question ("allergies?") grounds even when the interpreter says chitchat."""
    stub = sim_harness.StubLLM()
    sim_harness.install_stubs(monkeypatch, stub)
    from backend.voice_session import start_session
    session = start_session("foreign_body_airway", "resident-1")

    stub.intent_json = sim_harness._intent_json({"op": "ask_history", "target": "medications"})
    turn = sim_harness.run_turn(session, "What medications do you take?")
    assert turn.speaker == "patient"
    assert "medications" in session.engine.run.revealed_history
    assert "onset_while_eating" not in session.engine.run.revealed_history

    stub.intent_json = sim_harness._intent_json({"op": "chitchat"})
    turn = sim_harness.run_turn(session, "Do you have any allergies?", forced_addressee="patient")
    assert turn.speaker == "patient"
    assert "past_history_allergy" in session.engine.run.revealed_history
