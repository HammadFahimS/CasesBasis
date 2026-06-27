"""Director — proactive cue firing and speaker voice routing.

Extracted from :mod:`nurse_brain`.  Owns:

- **Speaker voice routing** — once the Router (:mod:`backend.routing`) has decided
  *who* is addressed, :func:`voice_reply` voices the right one (nurse vs
  EMS/family/patient/RT vs consult).
- **Proactive cue firing** — authored room prompts that come due mid-session.
- **Ask-back phrasing** — the line that names the available history-holders when the
  Router flags an ambiguous addressee.

Addressee resolution itself now lives in :mod:`backend.routing` (Room-of-Agents
C2); the old ``resolve_addressee`` chain was removed once the Router reached parity.
The one LLM touch-point (phrasing the nurse's reply) stays in
:mod:`nurse_brain.speak`; the Director calls it via a lazy import to avoid a
module-level circular dependency.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_ASK_BACK_PROMPT = "Whom should I direct that to?"


# ── Diagnostics boundary (C4) ─────────────────────────────────────────────────

def _announce_study(inv) -> str:
    """A deterministic 'it's ready, go read it' line for a visual study — never the
    interpretation. Templated off the study type so the LLM can't leak a read."""
    name = (inv.name or "study").strip()
    has_image = bool(getattr(inv, "image_ref", None) or getattr(inv, "visual_dx", None)) \
        or inv.category == "ecg"
    if inv.category == "ecg":
        return f"{name} is recorded — the tracing's up on the monitor for you to read."
    if has_image:
        return f"The {name} is up on the screen for you to review."
    return f"The {name} is back — the report's on the screen for you to read."


# ── Ask-back phrasing ─────────────────────────────────────────────────────────

def ask_back_prompt(engine) -> str:
    """Generate a spoken ask-back line that names the available providers."""
    from backend.actor_voice import provider_speaker
    seen: set[str] = set()
    names: list[str] = []
    for actor in engine.case.actors:
        sp = provider_speaker(actor.id)
        if sp == "nurse" or sp in seen:
            continue
        has_beats = any(provider_speaker(h.provider) == sp for h in engine.case.history)
        if has_beats:
            seen.add(sp)
            names.append(actor.role)
    if len(names) <= 1:
        return _ASK_BACK_PROMPT
    return f"Whom should I direct that to — {', '.join(names[:-1])}, or {names[-1]}?"


# ── Cue firing ────────────────────────────────────────────────────────────────

def fire_cues(session) -> list[dict]:
    """Voice any authored cues that came due this turn.

    Moved from :func:`nurse_brain._fire_cues`.  The Director owns proactive
    room events so they can later be gated by actor availability and state."""
    raw = session.engine.due_cues()
    if not raw:
        return []
    from backend.actor_voice import actor_speaker, voice_cue
    case = session.engine.case
    out: list[dict] = []
    for cue in raw:
        actor = next((a for a in case.actors if a.id == cue.get("actor_id")), None)
        speaker = actor_speaker(actor)
        line = voice_cue(session, actor, speaker, cue["text"])
        if not line:
            continue
        # Persist under the actor who actually spoke the cue, not a blanket "nurse"
        # (S2) — the transcript must record real identities.
        session.history.append({"role": speaker, "text": line})
        out.append({"speaker": speaker, "text": line})
    return out


# ── Voice routing ─────────────────────────────────────────────────────────────

def voice_reply(
    session,
    intent,
    result,
    vitals: dict,
    transcript: str,
    already_asked: bool = False,
) -> tuple[str, str]:
    """Decide who voices the reply and phrase it.

    Moved from :func:`nurse_brain._voice_reply`.

    - History from EMS / family / patient → spoken by that person.
    - Bedside respiratory measures → Respiratory Therapist (if present).
    - Everything else → nurse.

    Uses :func:`nurse_brain.speak` (lazy import, avoids circular dependency at
    module load time)."""
    from backend.nurse_brain import speak
    engine = session.engine
    speaker = "nurse"

    # History held by a bystander is voiced by them, not the nurse.
    if intent.op == "ask_history" and result is not None and result.revealed:
        beat = next((h for h in engine.case.history if h.id == intent.target), None)
        provider = beat.provider if beat else None
        from backend.actor_voice import provider_speaker, voice_history
        if provider_speaker(provider) != "nurse":
            reply, speaker = voice_history(
                session, provider, result.revealed, transcript,
                already_asked=already_asked,
            )
            logger.info("turn: op=%s target=%s provider=%s speaker=%s",
                        intent.op, intent.target, provider, speaker)
            return reply, speaker

    if intent.op == "order_investigation" and result is not None and result.revealed:
        inv = next((i for i in engine.case.investigations if i.id == intent.target), None)
        # Bedside measurements (VC/NIF) are reported as a number by the RT.
        if inv is not None and inv.category == "bedside":
            from backend.actor_voice import voice_bedside
            reply, rt_speaker = voice_bedside(session, result.revealed, transcript)
            if reply is not None:
                logger.info("turn: op=%s target=%s speaker=%s",
                            intent.op, intent.target, rt_speaker)
                return reply, rt_speaker
        # Visual studies (ECG, imaging) → announce availability ONLY, deterministically.
        # The interpretation belongs on the card for the resident to read; it must
        # never be voiced (C4) — and never routed through the embellishing LLM, which
        # would otherwise inject a read ("…showing sinus tach"). Card is built upstream.
        if inv is not None and inv.category in ("ecg", "imaging"):
            reply = _announce_study(inv)
            logger.info("turn: op=%s target=%s speaker=nurse (announce-only)",
                        intent.op, intent.target)
            return reply, speaker

    reply = speak(transcript, intent, result, vitals, session.history,
                  already_asked=already_asked, engine=engine)
    logger.info("turn: op=%s target=%s speaker=%s", intent.op, intent.target, speaker)
    return reply, speaker
