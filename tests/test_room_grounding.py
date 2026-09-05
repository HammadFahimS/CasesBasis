"""Room grounding (Phase 2 of the 2026-09-05 playtest fixes).

Four things the prod runs showed and this file guards:

* the nurse answered "is he deteriorating?" with "he's stable for now" over a
  patient in ``deteriorating`` — the chitchat brief carried no vitals;
* a re-asked patient beat, and any thanks/dismissal aimed at EMS, fell into the
  "I already told you" repeat path — the paramedic re-delivered the handover;
* the patient's free-roleplay exchanges never reached the room transcript;
* the drowsy register asked the patient to trail off mid-word, and gemma split
  words ("Food all ergies") that the TTS read as nonsense.
"""

from __future__ import annotations

import pytest

from backend import actor_voice, nurse_brain, patient_voice
from backend.emotional_state import for_speaker
from backend.nurse_brain import NurseIntent, _outcome_brief, run_turn
from backend.simulated_case_store import load_seed_case
from backend.simulation_engine import SimulationEngine
from tests import sim_harness


@pytest.fixture()
def db_path(tmp_path, monkeypatch):
    path = str(tmp_path / "room_grounding.db")
    monkeypatch.setenv("DB_PATH", path)
    monkeypatch.setenv("APP_SECRET", "test-secret-room-grounding")
    from backend.db import run_migrations
    from backend.simulated_case_repo import seed_gold_cases
    run_migrations()
    seed_gold_cases()
    yield path


def _fb_engine(state: str | None = None) -> SimulationEngine:
    eng = SimulationEngine(load_seed_case("foreign_body_airway"))
    if state:
        eng.goto(state)
    return eng


# ── 1 · the nurse answers a condition question from the monitor ───────────────

@pytest.mark.parametrize("line", [
    "Is he deteriorating?", "Nurse, how's he doing?", "Is he stable?",
    "Are his sats dropping?", "What's happening with him?", "Is he getting worse?",
    "How is the patient holding up?", "Is he okay?",
])
def test_condition_questions_are_recognised(line):
    assert nurse_brain._asks_about_condition(line)


@pytest.mark.parametrize("line", [
    "What do we have here?", "Catch me up.", "Tell me about the patient.",
    "Hi there.", "Let's get started.",
])
def test_open_prompts_are_not_condition_questions(line):
    assert not nurse_brain._asks_about_condition(line)


def test_condition_brief_carries_status_vitals_and_trend_when_deteriorating():
    eng = _fb_engine("deteriorating")                         # SpO₂ 84, BP 82/48, GCS 11
    vitals = eng.current_vitals().model_dump(exclude_none=True)
    brief = _outcome_brief(NurseIntent(op="chitchat"), None, vitals,
                           engine=eng, transcript="Is he deteriorating?")
    assert "Somnolent" in brief                               # the state's authored status
    assert "sats down from 97 to 84" in brief
    assert "blood pressure down from 138/84 to 82/48" in brief
    assert "GCS down from 15 to 11" in brief
    assert '"oxygen_saturation": 84' in brief
    assert "Never call him stable" in brief
    assert "hasn't given a specific order" not in brief       # not the bare invite


def test_condition_brief_says_unchanged_at_baseline():
    eng = _fb_engine()
    vitals = eng.current_vitals().model_dump(exclude_none=True)
    brief = _outcome_brief(NurseIntent(op="chitchat"), None, vitals,
                           engine=eng, transcript="How's he doing?")
    assert "unchanged since arrival" in brief
    assert '"oxygen_saturation": 97' in brief


def test_open_prompt_still_gets_the_invite_not_the_vitals():
    eng = _fb_engine("deteriorating")
    vitals = eng.current_vitals().model_dump(exclude_none=True)
    brief = _outcome_brief(NurseIntent(op="chitchat"), None, vitals,
                           engine=eng, transcript="What do we have here?")
    assert "hasn't given a specific order" in brief
    assert "84" not in brief                                  # vitals stay gated


def test_condition_question_reaches_the_speak_prompt_end_to_end(db_path, monkeypatch):
    from backend.voice_session import start_session
    seen = {}

    def fake_complete(system, user):
        if "routing brain" in system:
            return '{"op": "chitchat", "addressee": "nurse"}'
        seen["speak"] = user
        return "He's slipping, doctor. Sats are 84 and his pressure's 82 over 48."

    monkeypatch.setattr(nurse_brain, "_complete", fake_complete)
    session = start_session("foreign_body_airway", "resident-1")
    session.engine.goto("deteriorating")
    turn = run_turn(session, "Is he deteriorating?")
    assert turn.speaker == "nurse"
    assert "down from 97 to 84" in seen["speak"]
    assert session.history[-1] == {"role": "nurse", "text": turn.reply}


# ── 2 · already-told: the patient and the bystanders answer as people ─────────

def test_reasked_patient_beat_is_free_roleplay_on_the_transcript(db_path, monkeypatch):
    from backend.voice_session import start_session
    stub = sim_harness.StubLLM()
    sim_harness.install_stubs(monkeypatch, stub)
    prompts = []
    monkeypatch.setattr(patient_voice, "_complete", lambda s, u: prompts.append(u) or "Steak. At dinner.")
    session = start_session("foreign_body_airway", "resident-1")

    stub.intent_json = sim_harness._intent_json({"op": "ask_history", "target": "onset_while_eating"})
    first = run_turn(session, "What were you doing when this started?")
    assert first.speaker == "patient" and first.intent.op == "ask_history"
    assert "eating" in (first.revealed or "").lower()

    again = run_turn(session, "Sorry, what were you eating?")
    assert again.speaker == "patient"
    assert again.intent.op == "chitchat"                      # conversation, not a reveal
    assert again.revealed is None
    assert again.reply == "Steak. At dinner."
    assert "steak" in prompts[-1].lower()                     # the roleplay holds the facts
    # recorded on BOTH channels now — the room transcript used to miss it
    assert session.history[-2:] == [
        {"role": "resident", "text": "Sorry, what were you eating?"},
        {"role": "patient", "text": "Steak. At dinner."},
    ]
    assert session.patient_history[-1] == {"role": "patient", "text": "Steak. At dinner."}


def test_free_roleplay_smalltalk_lands_on_the_transcript(db_path, monkeypatch):
    from backend.voice_session import start_session
    stub = sim_harness.StubLLM()
    sim_harness.install_stubs(monkeypatch, stub)
    monkeypatch.setattr(patient_voice, "_complete", lambda s, u: "Hello doctor.")
    session = start_session("foreign_body_airway", "resident-1")
    stub.intent_json = sim_harness._intent_json({"op": "chitchat", "addressee": "patient"})
    turn = run_turn(session, "Hi, I'm the doctor.")
    assert turn.speaker == "patient"
    assert session.history[-1] == {"role": "patient", "text": "Hello doctor."}


def test_thanking_ems_after_the_handover_does_not_replay_it(db_path, monkeypatch):
    from backend.voice_session import start_session
    stub = sim_harness.StubLLM()
    sim_harness.install_stubs(monkeypatch, stub)
    prompts = []
    monkeypatch.setattr(actor_voice, "_complete", lambda s, u: prompts.append(u) or "No problem, doc. He's all yours.")
    session = start_session("foreign_body_airway", "resident-1")

    stub.intent_json = sim_harness._intent_json({"op": "ask_history", "target": "ems_handover"})
    handover = run_turn(session, "EMS, what happened?")
    assert handover.speaker == "ems" and "restaurant" in (handover.revealed or "").lower()

    stub.intent_json = sim_harness._intent_json({"op": "chitchat", "addressee": "ems"})
    thanks = run_turn(session, "Thanks guys, we've got him from here.")
    assert thanks.speaker == "ems"
    assert thanks.intent.op == "chitchat" and thanks.revealed is None
    assert thanks.reply == "No problem, doc. He's all yours."
    assert "already told" not in thanks.reply
    assert "What you have already told the resident" in prompts[-1]
    assert "restaurant" in prompts[-1]                        # he can still refer to it
    assert session.history[-1] == {"role": "ems", "text": thanks.reply}


def test_reasking_ems_is_a_reminder_not_the_repeat_path(db_path, monkeypatch):
    from backend.voice_session import start_session
    stub = sim_harness.StubLLM()
    sim_harness.install_stubs(monkeypatch, stub)
    prompts = []
    monkeypatch.setattr(actor_voice, "_complete", lambda s, u: prompts.append(u) or "[spoken reply]")
    session = start_session("foreign_body_airway", "resident-1")
    stub.intent_json = sim_harness._intent_json({"op": "ask_history", "target": "ems_handover"})
    run_turn(session, "EMS, what happened?")
    again = run_turn(session, "EMS, remind me, what happened again?")
    assert again.speaker == "ems" and again.intent.op == "chitchat"
    assert again.revealed is None
    assert "Acknowledge briefly that you already mentioned it" not in prompts[-1]   # the old repeat_note
    assert "never re-deliver the whole account" in prompts[-1]


def test_dismissing_ems_before_the_handover_is_still_conversation(db_path, monkeypatch):
    from backend.voice_session import start_session
    stub = sim_harness.StubLLM()
    sim_harness.install_stubs(monkeypatch, stub)
    session = start_session("foreign_body_airway", "resident-1")
    stub.intent_json = sim_harness._intent_json({"op": "chitchat", "addressee": "ems"})
    turn = run_turn(session, "Thanks, you can go.")
    assert turn.speaker == "ems" and turn.revealed is None
    assert "ems_handover" not in session.engine.run.revealed_history


def test_a_question_to_ems_with_a_thanks_in_it_still_takes_the_handover(db_path, monkeypatch):
    from backend.voice_session import start_session
    stub = sim_harness.StubLLM()
    sim_harness.install_stubs(monkeypatch, stub)
    session = start_session("foreign_body_airway", "resident-1")
    stub.intent_json = sim_harness._intent_json({"op": "chitchat", "addressee": "ems"})
    turn = run_turn(session, "Thanks for bringing him in, so what happened out there?")
    assert turn.speaker == "ems" and turn.intent.op == "ask_history"
    assert "restaurant" in (turn.revealed or "").lower()


def test_social_close_detection():
    assert nurse_brain._is_social_close("Thanks guys, we've got him from here.")
    assert nurse_brain._is_social_close("Cheers, you can go.")
    assert not nurse_brain._is_social_close("Thanks, what happened out there?")
    assert not nurse_brain._is_social_close("What happened?")


def test_bystander_chat_fails_soft_to_a_reminder(monkeypatch):
    from backend.voice_session import VoiceSession
    case = load_seed_case("foreign_body_airway")
    session = VoiceSession(id="s1", resident_id="r1", case=case, engine=SimulationEngine(case))
    monkeypatch.setattr(actor_voice, "_complete", lambda s, u: (_ for _ in ()).throw(RuntimeError("down")))
    ems = next(a for a in case.actors if a.id == "ems")
    told = [h.content for h in case.history if h.id == "ems_handover"]
    line = actor_voice.voice_chat(session, ems, "ems", "What happened again?", told)
    assert line.startswith("Like I said, doctor")
    assert "restaurant" in line
    from backend.agents import refuse
    assert actor_voice.voice_chat(session, ems, "ems", "Thanks.", []) == refuse("ems", ems)


# ── 3 · the patient never speaks in fragments ─────────────────────────────────

def test_drowsy_register_asks_for_whole_words_never_trailing_off():
    eng = _fb_engine("deteriorating")
    drowsy = for_speaker("patient", eng).instruction
    assert "four words" in drowsy and "whole" in drowsy
    for banned in ("trail off", "mumbled", "mid-word"):
        assert banned not in drowsy
    eng.run.vitals_override.update(oxygen_saturation=91)
    eng.goto("baseline")
    eng.run.vitals_override.update(oxygen_saturation=91)
    breathless = for_speaker("patient", eng).instruction
    assert "whole words" in breathless


def test_mend_rejoins_split_words_and_finishes_a_cut_off_word():
    vocab = patient_voice._case_vocab(load_seed_case("foreign_body_airway"))
    mend = patient_voice._mend_split_words
    assert mend("Food all ergies.", vocab) == "Food allergies."
    assert mend("No just blood press", vocab) == "No just blood pressure"
    # whole words, real short words and inflected stems are left alone
    for line in ("Told neck cancer chemo.", "Hard to swallow.", "I can't catch my breath.",
                 "Steak. At dinner.", "a way out", "It feels tight."):
        assert mend(line, vocab) == line
    assert mend("", vocab) == ""
    assert mend("Food all ergies.", frozenset()) == "Food all ergies."


def test_patient_reply_mends_a_split_word_before_it_is_spoken(monkeypatch):
    from backend.voice_session import VoiceSession
    case = load_seed_case("foreign_body_airway")
    session = VoiceSession(id="s1", resident_id="r1", case=case, engine=SimulationEngine(case))
    session.engine.goto("deteriorating")
    monkeypatch.setattr(patient_voice, "_complete", lambda s, u: "Food all... ergies.")
    assert patient_voice.patient_reply(session, "Any allergies?") == "Food allergies."
