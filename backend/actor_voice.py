"""Role-play a *grounded* bystander voice — EMS, family, the patient at handover.

When the resident takes a history beat (``ask_history``), the answer used to be
read flatly by the nurse. This module instead lets the person who actually holds
that information *speak it* — the paramedic gives the handover, the husband gives
the collateral — in character, but **grounded**: they convey the authored beat
content and must not invent beyond it. That's the difference from
:mod:`consult_voice` (a specialist who *advises*) and :mod:`patient_voice` (the
patient who free-form role-plays).

LLM access goes through the :func:`_complete` seam that tests monkeypatch.
"""

from __future__ import annotations

import logging

import re

from backend.actors import kind_for_word
from backend.patient_voice import _clean_for_speech, _complete

logger = logging.getLogger(__name__)


def provider_speaker(provider: str | None) -> str:
    """The speaker class that should voice a beat from this provider. A provider
    tag is a single word ('ems'/'family'/'patient'); anything unrecognised falls
    back to the nurse reading it."""
    kind = kind_for_word(provider)
    return kind.id if kind else "nurse"


def actor_speaker(actor) -> str:
    """The speaker class (styling + voice) for a case actor — the single source of
    truth shared by the session roster and the cue engine. Resolved from the
    actor's id/role against the cast registry, with consultants detected
    behaviourally and an unrecognised bystander treated as family."""
    if actor is None:
        return "nurse"
    from backend.consult_voice import _is_consultant
    for token in re.findall(r"[a-z']+", f"{actor.id} {actor.role}".lower()):
        kind = kind_for_word(token)
        if kind is not None:
            return kind.id
    if _is_consultant(actor):
        return "consult"
    return "family"


def find_actor(case, provider: str | None, speaker: str):
    """Best-effort: the case actor who embodies this provider (for persona/voice)."""
    prov = (provider or "").strip().lower()
    for actor in case.actors:
        tag = f"{actor.id} {actor.role}".lower()
        if prov and (prov in tag):
            return actor
    # fall back to any actor whose role matches the speaker class (e.g. an EMS actor)
    for actor in case.actors:
        if speaker in f"{actor.id} {actor.role}".lower():
            return actor
    return None


VOICE_SYSTEM = (
    "You are role-playing a real person at the bedside in an emergency-department "
    "simulation — not a clinician giving analysis. Stay fully in character; never "
    "mention being an AI or a simulation. You are handed the exact information you "
    "know and are conveying to the resident: say it aloud naturally, in your own "
    "voice and words, and do NOT add facts, numbers or opinions beyond what you "
    "were given. No markdown or stage directions — your words are read aloud. Keep "
    "it under about 60 words."
)


def _persona(actor, speaker: str) -> str:
    if actor is None:
        label = "the paramedic" if speaker == "ems" else "a family member"
        return f"You are {label} speaking to the resident."
    bits = [f"You are the {actor.role or actor.id}, speaking to the resident."]
    if actor.persona:
        bits.append(actor.persona)
    return "\n".join(bits)


def _voice_as(actor, speaker: str, content: str, transcript: str) -> str:
    """Role-play ``actor`` saying the grounded ``content`` aloud. Fails soft."""
    user = (
        f"{_persona(actor, speaker)}\n\n"
        f"The resident said: \"{transcript}\"\n\n"
        f"What you know and are telling them:\n{content}\n\nYou:"
    )
    try:
        return _clean_for_speech(_complete(VOICE_SYSTEM, user))
    except Exception as exc:  # never let role-play break the turn
        logger.warning("actor voice failed (%s): %s", speaker, exc)
        return _clean_for_speech(content)


def voice_history(session, provider: str | None, content: str, transcript: str) -> tuple[str, str]:
    """Voice an authored history beat as the person who holds it.

    Returns ``(reply, speaker)``. ``speaker`` is the styling/voice class
    (``ems``/``family``/``patient``/``nurse``). Grounded in ``content`` — the
    person paraphrases it aloud without inventing.
    """
    speaker = provider_speaker(provider)
    if speaker == "nurse" or not content:
        return content or "", "nurse"
    actor = find_actor(session.engine.case, provider, speaker)
    return _voice_as(actor, speaker, content, transcript), speaker


CUE_SYSTEM = (
    "You are a person at the bedside in an emergency-department simulation. You are "
    "given a stage direction describing a prompt you should give the resident right "
    "now (the team is lingering). Say it ALOUD, naturally and briefly, in your own "
    "voice — never read the stage direction verbatim or describe yourself in the "
    "third person. No markdown or stage directions. Under about 30 words."
)


def voice_cue(session, actor, speaker: str, text: str) -> str:
    """Turn an authored cue (a stage direction) into a natural spoken prompt from
    the actor. Fails soft to a lightly-cleaned version of the cue text."""
    user = f"{_persona(actor, speaker)}\n\nStage direction (say this in your own words): {text}\n\nYou:"
    try:
        return _clean_for_speech(_complete(CUE_SYSTEM, user))
    except Exception as exc:
        logger.warning("cue voice failed (%s): %s", speaker, exc)
        return _clean_for_speech(text)


def voice_bedside(session, content: str, transcript: str) -> tuple[str | None, str]:
    """Voice a bedside measurement (VC/NIF) as the Respiratory Therapist, if the
    case has one. Returns ``(None, "nurse")`` to signal 'let the nurse report it'."""
    if not content:
        return None, "nurse"
    actor = next(
        (a for a in session.engine.case.actors
         if a.id.lower() == "rt" or "respiratory" in a.role.lower()),
        None,
    )
    if actor is None:
        return None, "nurse"
    return _voice_as(actor, "rt", content, transcript), "rt"
