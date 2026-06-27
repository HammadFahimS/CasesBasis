"""Role-play a specialist the resident pages in a simulated case.

The mirror image of :mod:`patient_voice`, but for the consultants a case defines
as actors (neurology, ICU, respiratory therapy, …). The resident asks the nurse
to "call neurology"; the consultant calls back with a short, grounded
recommendation that also lands as a card in the Consults pane.

Like the patient, the consultant is *role-play* — it is NOT engine-grounded and
must not invent investigation results. It only advises on what it is told. LLM
access goes through the :func:`_complete` seam that tests monkeypatch.
"""

from __future__ import annotations

import logging

from backend.constitution import constitution
from backend.patient_voice import _clean_for_speech
from backend.sim_diagnostics import build_consult_card
from backend.voice_session import VoiceSession

logger = logging.getLogger(__name__)

CONSULT_SYSTEM = constitution("consult")

# Actors that are *not* consultants (they're the patient, the nurse, prehospital
# or family). Anything else a case lists is someone the resident could phone.
_NON_CONSULT = (
    "patient", "nurse", "ems", "paramedic", "medic", "husband", "wife", "mother",
    "father", "son", "daughter", "family", "partner", "relative", "bystander",
    "friend", "carer", "caregiver",
)


def _complete(system_prompt: str, user_prompt: str) -> str:
    """The consultant LLM seam (monkeypatched in tests)."""
    from backend.agent_runtime import complete

    return complete(system_prompt, user_prompt, kind="consult")


def _is_consultant(actor) -> bool:
    tag = f"{actor.id} {actor.role}".lower()
    return not any(word in tag for word in _NON_CONSULT)


def consultants(case) -> list:
    """The case's pageable specialists (everyone who isn't patient/nurse/EMS/family)."""
    return [a for a in case.actors if _is_consultant(a)]


def find_consultant(case, target: str | None):
    """Resolve a consult target (id or role text) to an actor, best-effort."""
    pool = consultants(case)
    if not pool:
        return None
    if not target:
        return pool[0]
    t = target.strip().lower()
    for a in case.actors:                       # exact id
        if a.id.lower() == t:
            return a
    for a in case.actors:                       # role text overlap
        role = a.role.lower()
        if role and (t in role or role in t):
            return a
    for a in pool:                              # any shared word
        if any(w and (w in a.role.lower() or w in a.id.lower()) for w in t.split()):
            return a
    return pool[0]


def _persona(actor) -> str:
    label = actor.role or actor.id
    bits = [f"You are the {label} being consulted on this patient."]
    if actor.persona:
        bits.append(actor.persona)
    if actor.script:
        bits.append("Things you might say: " + " | ".join(actor.script))
    return "\n".join(bits)


def _case_brief(session: VoiceSession) -> str:
    case = session.engine.case
    p = case.patient
    state = session.engine.state
    lines = [
        f"Patient: {p.name or 'the patient'}, a {p.age}-year-old {p.gender}.",
        f"Presenting complaint: {case.presentation.chief_complaint}.",
    ]
    if p.past_medical_history:
        lines.append("Past medical history: " + ", ".join(p.past_medical_history) + ".")
    if p.medications:
        lines.append("Medications: " + ", ".join(p.medications) + ".")
    if state.patient_status:
        lines.append(f"Current status: {state.patient_status}.")
    return "\n".join(lines)


def _recent(history: list[dict], limit: int = 6) -> str:
    out = []
    for turn in history[-limit:]:
        who = "Resident" if turn.get("role") == "resident" else "Nurse"
        line = (turn.get("text") or "").strip()
        if line:
            out.append(f"{who}: {line}")
    return "\n".join(out)


def consult_reply(session: VoiceSession, target: str | None, transcript: str) -> tuple[str, dict | None]:
    """The consultant's spoken recommendation plus a Consults-pane card.

    Returns ``(reply, card)``. History is left to the caller (``run_turn``) so the
    exchange is recorded exactly once on the nurse channel."""
    case = session.engine.case
    actor = find_consultant(case, target)
    if actor is None:
        return ("There's no specialist available to take that consult right now.", None)

    label = actor.role or actor.id
    user = (
        f"{_persona(actor)}\n\n"
        f"Case so far:\n{_case_brief(session)}\n\n"
        f"Recent bedside exchange:\n{_recent(session.history) or '(none)'}\n\n"
        f'The resident pages you and says: "{transcript}"\n{label}:'
    )
    try:
        reply = _clean_for_speech(_complete(CONSULT_SYSTEM, user))
    except Exception as exc:  # never let a consult break the turn
        logger.warning("consult reply failed: %s", exc)
        reply = (
            f"This is {label}. I can't give detailed advice right now — keep "
            "supporting the airway and breathing, and call me back with an update."
        )
    return reply, build_consult_card(label, reply, actor.id)
