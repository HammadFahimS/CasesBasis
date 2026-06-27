"""Room-of-Agents C3 — the room's character agents and the no-fabrication contract.

The Router (:mod:`backend.routing`) decides **who** answers a turn; the existing
voice seams (:func:`nurse_brain.speak`, :mod:`backend.actor_voice`,
:mod:`backend.patient_voice`, :mod:`backend.consult_voice`) decide **how** that one
phrases it. This module makes the C3 contract explicit and gives it its one missing
capability:

- **Every actor is bounded to its own knowledge** and **never speaks for another.**
- When a *present* actor is addressed about something outside what they hold, they
  **refuse in character** (:func:`refuse`) — the nurse no longer fabricates a cover.
  This removes the last residue of the nurse-as-sink (e.g. asking the Respiratory
  Therapist for past medical history now gets an honest "not my area", not a nurse
  guess).

``ROOM_AGENTS`` documents what each kind is allowed to answer from. The full
composition is ``CharacterAgent = Constitution + CaseFlavor + EmotionalState +
VoiceBinding``:

- **Constitution** — invariant rules per kind (:mod:`backend.constitution`).
- **CaseFlavor** — the per-case persona/script (data on the actor; composed in each
  voice module's user prompt).
- **EmotionalState** — live, per-turn, from engine vitals/state
  (:func:`backend.emotional_state.for_speaker`): the patient/relative escalate, the
  professionals stay steady; drives both a prompt instruction and a TTS stability.
- **VoiceBinding** — the ElevenLabs/Together voice for the kind
  (:meth:`backend.actors.SpeakerKind.voice`).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Speaker kind → the knowledge it is bounded to. The Router + voice seams jointly
# enforce this; nothing here invents clinical content (the engine stays the single
# source of truth).
ROOM_AGENTS: dict[str, str] = {
    "nurse": "bedside coordinator — engine outcomes, order read-backs, vitals",
    "patient": "first-person symptoms and authored patient history beats",
    "ems": "the prehospital handover beat(s)",
    "family": "collateral history beat(s) about the patient",
    "rt": "bedside respiratory measurements (VC/NIF), on order",
    "consult": "specialist advice, when called",
}

# In-character declines for a present actor with nothing grounded to offer. Kept
# deterministic (no LLM): an honest fixed line is more reliable than a model call on
# a rare path, and it can never leak an invented clinical detail.
_REFUSALS: dict[str, str] = {
    "ems": "That's beyond what we got on scene, doctor — I've handed over everything from the field.",
    "family": "I'm sorry, I wouldn't know about that — you'd have to ask her doctors.",
    "rt": "That's outside my area — I'm here for the airway and breathing. Best ask the team.",
    "patient": "I'm not sure about that, doctor.",
    "consult": "That's outside what I can tell you from here — I'd defer to the primary team.",
}


def refuse(speaker: str, actor=None) -> str:
    """An in-character 'that's not something I'd know' from a present actor bounded
    away from what was asked — so the nurse never covers for them (C3)."""
    line = _REFUSALS.get(speaker)
    if line:
        return line
    who = getattr(actor, "role", None) or "I"
    return f"I'm sorry — that's not something {('I' if who == 'I' else who)} would know."
