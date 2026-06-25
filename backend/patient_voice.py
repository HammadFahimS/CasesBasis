"""Role-play the patient's voice in a simulated case.

Distinct from the nurse: the nurse executes clinical commands and is grounded in
the engine; the patient simply speaks in character (history-taking, like the
manual simulator's patient Q&A). The resident addresses the patient by default —
the nurse only when the utterance begins with 'nurse'.

LLM access goes through the :func:`_complete` seam that tests monkeypatch.
"""

from __future__ import annotations

import logging
import re

from backend.constitution import constitution
from backend.voice_session import VoiceSession

logger = logging.getLogger(__name__)

PATIENT_SYSTEM = constitution("patient")


def _clean_for_speech(text: str) -> str:
    """Strip pause-faking punctuation so the TTS reads the reply at a natural pace.

    Models render breathlessness as ``I...feel...short...of...breath`` with em-dash
    line breaks; the TTS turns every run of dots/dashes into a hard pause. Collapse
    those into ordinary spacing while leaving real punctuation and hyphenated words
    (``x-ray``, ``follow-up``) intact.
    """
    text = text.replace("…", "...")
    text = re.sub(r"(?:\.\s*){2,}", " ", text)      # "..." / ". . ." → space
    text = re.sub(r"\s*[—–]\s*", " ", text)         # em/en dash separators → space
    text = re.sub(r"\s+-+\s+", " ", text)           # spaced/duplicated hyphens → space
    text = re.sub(r"\s+", " ", text)                # collapse newlines & runs of space
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)    # no space before punctuation
    return text.strip()


def _complete(system_prompt: str, user_prompt: str) -> str:
    """The patient/bystander LLM seam (monkeypatched in tests). Reused by
    :mod:`backend.actor_voice`; delegates to the shared interactive runtime."""
    from backend.agent_runtime import complete

    return complete(system_prompt, user_prompt, kind="patient")


def _persona(case) -> str:
    p = case.patient
    bits = [f"You are {p.name or 'the patient'}, a {p.age}-year-old {p.gender}."]
    if p.persona:
        bits.append(p.persona)
    actor = next(
        (a for a in case.actors if a.id == "patient" or a.role.lower() == "patient"),
        None,
    )
    if actor:
        if actor.persona:
            bits.append(actor.persona)
        if actor.script:
            bits.append("Things you might say: " + " | ".join(actor.script))
    if p.past_medical_history:
        bits.append("Your past medical history: " + ", ".join(p.past_medical_history) + ".")
    if p.medications:
        bits.append("Your medications: " + ", ".join(p.medications) + ".")
    if p.allergies:
        bits.append("Your allergies: " + ", ".join(p.allergies) + ".")
    known = [h.content for h in case.history if h.provider == "patient" and h.content]
    if known:
        bits.append(
            "Background facts about your illness you may draw on — but ONLY reveal a "
            "fact when the doctor's question actually touches on it; never recite "
            "this list or dump it all at once: " + " ".join(known)
        )
    bits.append(f"Your main complaint, if asked what's wrong: {case.presentation.chief_complaint}.")
    return "\n".join(bits)


# States in which the patient is sedated/unresponsive and cannot speak.
_SILENT_STATES = {"post_intubation", "pea_arrest", "icu_admission"}


def _affect_stability(engine) -> float:
    """ElevenLabs stability for the patient this turn — lower = more affect/expression.

    Mirrors the emotional escalation in :func:`_affect_instruction`: calm at
    baseline, increasingly expressive (lower stability) as the patient tires."""
    state_id = engine.run.current_state_id
    if state_id in _SILENT_STATES:
        return 0.30   # moot — no speech; keep the default
    if "intubation" in state_id and "post" not in state_id:
        return 0.15   # near-failure: maximum affect, almost no stability
    elapsed = engine.elapsed_in_state
    if elapsed > 180:
        return 0.20
    if elapsed > 60:
        return 0.25
    return 0.30   # baseline breathless anxiety


def _affect_instruction(engine) -> str:
    """A behavioral modifier injected into the patient's turn prompt.

    Escalates distress as the patient deteriorates — shorter replies, more
    breathless language, eventual silence when sedated. Derived from the
    engine's current state and time elapsed in that state, so it generalizes
    to any case whose states follow the same patient-status pattern."""
    state_id = engine.run.current_state_id
    if state_id in _SILENT_STATES:
        return (
            "You are sedated and cannot speak. Do not reply with words — "
            "respond with silence (output an empty string)."
        )
    elapsed = engine.elapsed_in_state
    # intubation_prep: respiratory muscles nearly failing
    if "intubation" in state_id and "post" not in state_id:
        return (
            "You are exhausted and barely able to breathe. Reply in at most "
            "four words. Every breath is a visible effort — do not form full "
            "sentences. You may trail off mid-word."
        )
    # Escalate within baseline/recognition as time passes
    if elapsed > 180:
        return (
            "Your breathing has worsened over the last few minutes. Keep each "
            "reply to one very short phrase (≤6 words). Show increasing "
            "exhaustion and breathlessness."
        )
    if elapsed > 60:
        return (
            "You are noticeably more breathless than when the doctor arrived. "
            "Limit replies to short sentences and convey effort with each word."
        )
    return (
        "You are breathless and anxious but can still speak in short sentences. "
        "You cannot complete full sentences without pausing to breathe."
    )


def _recent(history: list[dict], limit: int = 6) -> str:
    out = []
    for turn in history[-limit:]:
        who = "Doctor" if turn.get("role") == "resident" else "You"
        line = (turn.get("text") or "").strip()
        if line:
            out.append(f"{who}: {line}")
    return "\n".join(out)


def patient_reply(session: VoiceSession, transcript: str) -> str:
    """The patient's spoken reply to the resident (role-play, not engine-grounded)."""
    case = session.engine.case
    status = session.engine.state.patient_status
    affect = _affect_instruction(session.engine)
    user = (
        f"{_persona(case)}\n\n"
        f"Right now: {status or 'you are being assessed in the emergency department.'}\n\n"
        f"How you are speaking right now: {affect}\n\n"
        f"Recent conversation:\n{_recent(session.patient_history) or '(none)'}\n\n"
        f"Doctor: {transcript}\nYou:"
    )
    # Sedated/unresponsive states: don't burn an LLM call on silence.
    if session.engine.run.current_state_id in _SILENT_STATES:
        return ""
    try:
        reply = _clean_for_speech(_complete(PATIENT_SYSTEM, user))
    except Exception as exc:
        logger.warning("patient reply failed: %s", exc)
        reply = "I'm sorry, I can't really talk right now."
    session.patient_history.append({"role": "resident", "text": transcript})
    session.patient_history.append({"role": "patient", "text": reply})
    return reply
