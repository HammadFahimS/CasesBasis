"""Director — the single seam that decides 'who speaks this turn?'.

Extracted from :mod:`nurse_brain`.  Owns:

- **Addressee resolution** — explicit vocative → forced chip selection → ask-back
  when confidence is too low.
- **Speaker voice routing** — nurse vs EMS/family/patient/RT vs consult.
- **Proactive cue firing** — authored room prompts that come due mid-session.

The Director is ~80 % deterministic.  The one LLM touch-point (phrasing the
nurse's reply) stays in :mod:`nurse_brain.speak`; the Director calls it via a
lazy import to avoid a module-level circular dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_ASK_BACK_PROMPT = "Whom should I direct that to?"


# ── Decision dataclass ────────────────────────────────────────────────────────

@dataclass
class RouteDecision:
    """The Director's verdict for one turn."""
    ask_back: bool = False           # unsure who this is for → ask the resident
    forced_speaker: str | None = None  # honour a chip tap


# ── Addressee resolution ──────────────────────────────────────────────────────

def resolve_addressee(
    intent,
    engine,
    forced_addressee: str | None = None,
) -> RouteDecision:
    """Return the routing verdict for this turn.

    Priority:
    1. ``forced_addressee`` (user tapped an addressee chip) — trusted completely.
    2. Ask-back: interpreter flagged uncertainty on an ask_history turn AND the
       case has 2+ history-holding providers (patient + at least one bystander).
    3. No override — existing :func:`nurse_brain._apply_addressee` handles it.
    """
    if forced_addressee:
        return RouteDecision(forced_speaker=forced_addressee)
    if _should_ask_back(intent, engine):
        return RouteDecision(ask_back=True)
    return RouteDecision()


def _should_ask_back(intent, engine) -> bool:
    """True only when the Director is genuinely unsure who the question is for.

    Conservative by design — only fires when the interpreter itself expressed
    uncertainty (clarify=True) on a history-taking turn, AND the case roster
    has 2+ providers who could plausibly answer."""
    if not getattr(intent, "clarify", False):
        return False
    if getattr(intent, "op", None) != "ask_history":
        return False
    from backend.actor_voice import provider_speaker
    providers = {provider_speaker(h.provider) for h in engine.case.history}
    providers.discard("nurse")
    return len(providers) >= 2


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
        session.history.append({"role": "nurse", "text": line})
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

    # Bedside measurements are reported by the RT (if the case has one).
    if intent.op == "order_investigation" and result is not None and result.revealed:
        inv = next((i for i in engine.case.investigations if i.id == intent.target), None)
        if inv is not None and inv.category == "bedside":
            from backend.actor_voice import voice_bedside
            reply, rt_speaker = voice_bedside(session, result.revealed, transcript)
            if reply is not None:
                logger.info("turn: op=%s target=%s speaker=%s",
                            intent.op, intent.target, rt_speaker)
                return reply, rt_speaker

    reply = speak(transcript, intent, result, vitals, session.history,
                  already_asked=already_asked)
    logger.info("turn: op=%s target=%s speaker=%s", intent.op, intent.target, speaker)
    return reply, speaker
