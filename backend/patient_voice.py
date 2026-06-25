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

from backend.voice_session import VoiceSession

logger = logging.getLogger(__name__)

PATIENT_SYSTEM = (
    "You are role-playing a patient in an emergency-department simulation. Stay "
    "fully in character as this one person — never break role, never mention being "
    "an AI or a simulation, and never clinically analyse yourself.\n"
    "Answer in the first person like a real patient at triage: ANSWER ONLY THE "
    "QUESTION YOU WERE ASKED, in one or two short sentences. Do NOT recite your "
    "whole story, list all your symptoms, or volunteer details the doctor hasn't "
    "asked about yet — make them ask. If a question is open ('what's going on?'), "
    "give just your main complaint in a sentence and stop; let the doctor probe "
    "for the rest. Mention a specific symptom only when they ask about that area. "
    "It is good to be brief — a real frightened, breathless patient does not give "
    "speeches.\n"
    "Reflect how you feel right now: if you are breathless or frightened, keep "
    "your words short and simple, but speak in complete sentences. Show distress "
    "through what you say, not through punctuation — never break words apart with "
    "ellipses ('...'), dashes, or line breaks; your reply is read aloud and those "
    "create unnatural stop-start pauses. Only share what this patient would "
    "plausibly know; do not invent major new medical facts beyond your "
    "background. No markdown or stage directions — your words are spoken aloud."
)


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
    from agents.base import build_llm_client, get_llm_provider, resolve_model

    provider = get_llm_provider()
    client = build_llm_client(provider)
    model = resolve_model(provider, None)
    return client.text_response(
        model=model, system_prompt=system_prompt, user_prompt=user_prompt
    ).strip()


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
    user = (
        f"{_persona(case)}\n\n"
        f"Right now: {status or 'you are being assessed in the emergency department.'}\n\n"
        f"Recent conversation:\n{_recent(session.patient_history) or '(none)'}\n\n"
        f"Doctor: {transcript}\nYou:"
    )
    try:
        reply = _clean_for_speech(_complete(PATIENT_SYSTEM, user))
    except Exception as exc:
        logger.warning("patient reply failed: %s", exc)
        reply = "I'm sorry, I can't really talk right now."
    session.patient_history.append({"role": "resident", "text": transcript})
    session.patient_history.append({"role": "patient", "text": reply})
    return reply
