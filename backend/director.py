"""Director — proactive cue firing and speaker voice routing.

Presented to the resident as the **Head Physician**: when the Director/Router can't
tell who a request is for, it speaks to the resident in that identity (a distinct
authoritative voice, speaker class ``head``) to ask them to clarify — rather than
guessing and dropping the utterance on the patient.

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

def _announce_study(inv, image: str | None = None) -> str:
    """A deterministic 'it's ready, go read it' line for a visual study — never the
    interpretation. Templated off the study type so the LLM can't leak a read.
    ``image`` is the authored picture resolved for the current state, if any."""
    name = (inv.name or "study").strip()
    has_image = bool(image or getattr(inv, "image_ref", None) or getattr(inv, "visual_dx", None)) \
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


def head_clarify_prompt(engine) -> str:
    """The Head Physician's spoken line when a request's addressee is genuinely unclear.

    Names a couple of people actually in the room so the resident has something concrete
    to aim at, and asks them to be more specific or to name who they want."""
    from backend.actor_voice import actor_speaker
    people: list[str] = []
    for a in engine.case.actors:
        if actor_speaker(a) == "nurse" or not a.role:
            continue
        try:
            present = engine.is_present(a.id)
        except Exception:
            present = True
        if present:
            people.append(a.role)
    opts = ["the patient", "the nurse"] + people[:3]
    listed = f"{', '.join(opts[:-1])}, or {opts[-1]}"
    return ("Head physician here — I'm not sure who that was meant for. Could you say it "
            f"again with a bit more detail, or tell me who you want: {listed}?")


# ── Cue firing ────────────────────────────────────────────────────────────────

def fire_cues(session) -> list[dict]:
    """Voice any authored cues that came due this turn, plus actor arrivals.

    Moved from :func:`nurse_brain._fire_cues`.  The Director owns proactive
    room events so they can be gated by actor availability and state. Arrivals
    fire deterministically the moment a timed actor enters the room (C1) — the
    nurse announces "her husband's just arrived" *when he actually arrives*, not
    on a state-linger timer that could fire before he's here."""
    engine = session.engine
    out: list[dict] = []

    # Arrivals first — the room should register who's here before nudging the team.
    for arr in engine.due_arrivals():
        role = (arr.get("role") or "").strip() or "Someone"
        line = f"{role}'s just arrived and is at the bedside."
        session.history.append({"role": "nurse", "text": line})
        out.append({"speaker": "nurse", "text": line})

    raw = engine.due_cues()
    if raw:
        from backend.actor_voice import actor_speaker, voice_cue
        case = engine.case
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


def pending_deferred(session) -> list[dict]:
    """The set of ordered results still in flight (scheduled, not yet back).

    Lets the room show the holding actor as *working* — a spinner on their seat in
    the round-table view — from the moment they acknowledge the order until the
    value lands and they report it (the counterpart to :func:`fire_deferred`, which
    fires the due ones). Each row is ``{speaker, label, wait_seconds}`` keyed by the
    holder's speaker class so the front-end can light the right seat. Call *after*
    :func:`fire_deferred` so anything that came due this turn is already drained."""
    engine = session.engine
    events = getattr(engine.run, "deferred_events", None) or []
    if not events:
        return []
    from backend.actor_voice import actor_speaker
    out: list[dict] = []
    for ev in events:
        inv = next((i for i in engine.case.investigations if i.id == ev.get("inv_id")), None)
        if inv is None:
            continue
        actor = next((a for a in engine.case.actors if a.id == ev.get("actor_id")), None)
        speaker = actor_speaker(actor) if actor else "nurse"
        wait = max(0, int(ev.get("fire_at", 0)) - engine.run.elapsed_seconds)
        out.append({"speaker": speaker, "label": inv.name, "wait_seconds": wait})
    return out


def fire_deferred(session) -> list[dict]:
    """Surface any scheduled results that have come back (the room talking back).

    For each due deferred result the holding actor (RT for bedside spirometry, the
    nurse for sent labs) reports the number aloud and a card lands in the diagnostics
    panel. Deterministic, terse, signal-only — the same C4 discipline as a live read:
    the actor states the value, never the interpretation. Returns ``{speaker, text,
    card}`` rows so both the turn payload and the live WS push can render them."""
    engine = session.engine
    due = engine.due_deferred()
    if not due:
        return []
    from backend.actor_voice import actor_speaker
    from backend.patient_voice import _clean_for_speech
    from backend.sim_diagnostics import build_investigation_card
    out: list[dict] = []
    state_id = engine.run.current_state_id
    for item in due:
        inv = item["inv"]
        revealed = item["revealed"] or ""
        actor = next((a for a in engine.case.actors if a.id == item.get("actor_id")), None)
        speaker = actor_speaker(actor) if actor else "nurse"
        image = inv.image_for(state_id)
        # A result that is a picture is announced, never described (C4).
        line = _announce_study(inv, image) if image \
            else _clean_for_speech(f"{inv.name} is back — {revealed}.")
        card = build_investigation_card(
            inv, engine.current_vitals(), engine.case.patient, revealed, state_id=state_id
        )
        session.history.append({"role": speaker, "text": line})
        session.cards.append(card)
        out.append({"speaker": speaker, "text": line, "card": card})
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

    if intent.op == "order_investigation" and result is not None:
        inv = next((i for i in engine.case.investigations if i.id == intent.target), None)
        # A study that resolves to an authored picture (a scope view, a film) is
        # announce-only, whatever its category: the image is what the resident
        # reads (C4). The study's result holder voices the announce — the RT for a
        # bedside procedure — so the "view's up" lands in the right voice.
        image = inv.image_for(engine.run.current_state_id) if inv is not None else None
        if image and result.revealed:
            from backend.actor_voice import actor_speaker
            holder = next((a for a in engine.case.actors if a.id == inv.result_actor_id), None)
            speaker = actor_speaker(holder) if holder else "nurse"
            reply = _announce_study(inv, image)
            logger.info("turn: op=%s target=%s speaker=%s (announce-only, image)",
                        intent.op, intent.target, speaker)
            return reply, speaker
        # Bedside measurements (VC/NIF) are owned by the RT, never the nurse — whether
        # the value is in hand now (``revealed``) OR the order has a turn-around and the
        # RT is acknowledging it (``message`` from the deferred-result path: "on it,
        # back in about a minute"). The RT voicing the ack is what makes the later
        # report-back land in the same voice.
        bedside_content = result.revealed or result.message
        if inv is not None and inv.category == "bedside" and bedside_content:
            from backend.actor_voice import voice_bedside
            reply, rt_speaker = voice_bedside(session, bedside_content, transcript)
            if reply is not None:
                logger.info("turn: op=%s target=%s speaker=%s",
                            intent.op, intent.target, rt_speaker)
                return reply, rt_speaker
        # Visual studies (ECG, imaging) → announce availability ONLY, deterministically.
        # The interpretation belongs on the card for the resident to read; it must
        # never be voiced (C4) — and never routed through the embellishing LLM, which
        # would otherwise inject a read ("…showing sinus tach"). Card is built upstream.
        if inv is not None and inv.category in ("ecg", "imaging") and result.revealed:
            reply = _announce_study(inv)
            logger.info("turn: op=%s target=%s speaker=nurse (announce-only)",
                        intent.op, intent.target)
            return reply, speaker

    reply = speak(transcript, intent, result, vitals, session.history,
                  already_asked=already_asked, engine=engine)
    logger.info("turn: op=%s target=%s speaker=%s", intent.op, intent.target, speaker)
    return reply, speaker
