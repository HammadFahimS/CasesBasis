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


_WORD_RE = re.compile(r"[A-Za-z']+")
# Inflections a truncated word must NOT be "completed" with: "swallow" is a whole
# word even when only "swallowing" is in the case vocabulary.
_INFLECTIONS = frozenset({
    "s", "es", "ed", "ing", "ly", "er", "ers", "est", "ness", "less", "ful", "y",
    "ies", "ier", "iest", "ment", "able", "ible", "ion", "ions", "ive", "al",
})


def _case_vocab(case) -> frozenset[str]:
    """Every word the patient's own prompt already contains — the persona, the
    authored facts, the status lines, the constitution. The only words a mend may
    produce, so it can restore a word the model broke but never invent one."""
    text = " ".join([PATIENT_SYSTEM, _persona(case)]
                    + [(s.patient_status or "") for s in case.states])
    return frozenset(w.lower() for w in _WORD_RE.findall(text))


def _mend_split_words(text: str, vocab: frozenset[str]) -> str:
    """Re-join a word the model broke in two and finish one it cut off at the end.

    A patient told to sound breathless sometimes renders the effort as word
    fragments — ``Food all ergies``, ``No just blood press`` — which the TTS reads
    as nonsense. Two bounded repairs, both limited to words already in the
    patient's prompt: two adjacent tokens that concatenate to a known word (when at
    least one of them is not a word on its own) are joined; a final token that is
    not a known word but is the unique prefix of one (by more than an inflection)
    is completed. Anything else is left exactly as written."""
    if not vocab or not text:
        return text
    parts = re.findall(r"[A-Za-z']+|[^A-Za-z'\s]+|\s+", text)
    i = 0
    while i + 2 < len(parts):
        a, gap, b = parts[i], parts[i + 1], parts[i + 2]
        if (a[:1].isalpha() and gap.isspace() and b[:1].isalpha()
                and min(len(a), len(b)) >= 2 and len(a) + len(b) >= 6
                and (a.lower() + b.lower()) in vocab
                and (a.lower() not in vocab or b.lower() not in vocab)):
            parts[i:i + 3] = [a + b.lower()]
            continue
        i += 1
    last = next((k for k in range(len(parts) - 1, -1, -1) if parts[k][:1].isalpha()), None)
    if last is not None:
        t = parts[last]
        low = t.lower()
        if len(t) >= 4 and low not in vocab:
            cands = [w for w in vocab
                     if w.startswith(low) and len(w) >= len(low) + 2
                     and w[len(low):] not in _INFLECTIONS]
            if len(cands) == 1:
                parts[last] = t + cands[0][len(low):]
    return "".join(parts)


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
            "Facts about your illness — when the doctor asks about a symptom or timeline, "
            "give them a real answer drawn from these facts (in the register your condition "
            "dictates). Do NOT volunteer everything at once or dump the list; answer what "
            "is asked. But when asked 'how are you feeling' or 'tell me what's wrong', "
            "share the most prominent fact (your breathing, your weakness) — not 'I don't "
            "know'. Facts: " + " ".join(known)
        )
    bits.append(f"Your main complaint, if asked what's wrong: {case.presentation.chief_complaint}.")
    return "\n".join(bits)


# States in which the patient is sedated/unresponsive and cannot speak.
# The patient's live emotional model now lives in backend.emotional_state (Phase 5),
# generalised across the cast. These thin shims preserve the call sites here.
from backend.emotional_state import is_silent as _is_silent
from backend.emotional_state import for_speaker as _emotion_for


def _affect_stability(engine) -> float:
    """ElevenLabs stability for the patient this turn (delegates to emotional_state)."""
    return _emotion_for("patient", engine).stability


def _affect_instruction(engine) -> str:
    """The patient's behavioural modifier this turn (delegates to emotional_state)."""
    return _emotion_for("patient", engine).instruction


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
    if _is_silent(session.engine):
        return ""
    try:
        reply = _mend_split_words(_clean_for_speech(_complete(PATIENT_SYSTEM, user)),
                                  _case_vocab(case))
    except Exception as exc:
        logger.warning("patient reply failed: %s", exc)
        reply = "I'm sorry, I can't really talk right now."
    session.patient_history.append({"role": "resident", "text": transcript})
    session.patient_history.append({"role": "patient", "text": reply})
    return reply
