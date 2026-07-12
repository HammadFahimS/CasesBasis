"""The Router — Room-of-Agents C2 (docs/room-of-agents.md).

One spoken utterance → an explicit :class:`Route`: **who** is being addressed and
**whether they are in the room**, decided server-side from the interpreted intent
and the room's presence model. This replaces the old single-axis
``director.resolve_addressee`` chain whose only real fallback was the nurse, which
laundered every routing miss into confident filler.

The Router decides *who speaks*; it never invents clinical content. Grounding
stays in :mod:`simulation_engine` (the single source of truth) and the per-actor
agents/voices. The one upstream LLM call (``nurse_brain.interpret``) already
classified the speech-act; the Router consumes that — it does not add a call.

Resolution priority (first match wins):

1. **Forced chip** — the resident tapped an addressee; trusted.
2. **Pending answer** — an affirmation / dose / confirm while an action is parked
   stays on the staff channel, so a confirmation can't be spoken to the patient and
   loop forever (S3).
3. **Explicit vocative** — a leading "Husband, …" / "EMS, …" names a person over
   the *full* roster; presence is then checked.
4. **Summon** — "call / page / get the husband" brings an absent actor in.
5. **Speech-act** — request_consult → the consultant; ask_history → the beat's
   holder; staff orders → the nurse; bare conversation → the patient.

A resolved addressee who is **not present** yields ``Route(present=False)`` so the
turn can answer honestly ("he isn't here yet") instead of falling to the nurse.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from backend.actors import kind_for_word

logger = logging.getLogger(__name__)

# Staff speech-acts the nurse owns (orders, tests, exams, diagnosis, vitals).
_STAFF_OPS = {
    "order_medication", "order_investigation", "perform_exam",
    "perform_action", "diagnose", "read_vitals",
}

# "Bring someone into the room" verbs — used to read a summon out of an utterance
# that names an absent person ("call the husband", "page neurology", "get her son").
_SUMMON_RE = re.compile(
    r"\b(call|phone|page|bug|summon|get|grab|bring|fetch|find|ask for|send for|"
    r"go (?:get|find|talk to|speak)|wake)\b",
    re.IGNORECASE,
)


@dataclass
class Route:
    """The Router's verdict for one turn.

    ``addressee`` is a speaker *kind* (``nurse``/``patient``/``ems``/``family``/
    ``rt``/``consult``) for voicing; ``actor_id`` is the concrete case actor when one
    was resolved (carries presence + persona). ``present`` is False when the
    addressee is a real person not currently in the room."""
    addressee: str                       # speaker kind that should answer
    speech_act: str                      # normalised op (== intent.op)
    actor_id: str | None = None          # concrete case actor, when resolved
    present: bool = True                 # is the addressee in the room?
    summon: str | None = None            # actor id to bring in this turn
    ask_back: bool = False               # Head Physician is unsure — ask the resident
    reason: str = ""                     # why this route fired (trace + tests)


# ── public entry point ────────────────────────────────────────────────────────

def route(intent, engine, transcript: str = "", forced_addressee: str | None = None) -> Route:
    """Resolve the addressee for one already-interpreted utterance. See module docs."""
    op = getattr(intent, "op", "chitchat")

    # 1 — forced chip: trusted, but still resolve its concrete actor for voicing.
    if forced_addressee:
        if forced_addressee in ("patient", "nurse"):
            return Route(forced_addressee, op, reason="forced chip")
        actor = _actor_for_kind_or_id(engine, forced_addressee)
        kind = _kind_of(engine, actor) or forced_addressee
        return Route(kind, op, actor_id=actor.id if actor else None,
                     present=True, reason="forced chip")

    # 2 — answer to a parked confirmation stays on the staff channel (S3).
    if _resolves_pending(intent, transcript, engine):
        return Route("nurse", op, reason="resolves pending action")

    # 3 — "call / page / get <absent person>" — the nurse fetches them. Detected
    # before vocative so "Nurse, call the husband" summons (nurse acknowledges)
    # rather than the husband being addressed directly.
    summon_id = _summon_target(transcript, engine)
    if summon_id is not None:
        return Route("nurse", op, actor_id=summon_id, summon=summon_id, reason="summon")

    # 4 — explicit vocative names a person over the full roster.
    voc = _leading_vocative(transcript)
    if voc is not None:
        decided = _route_to_kind(engine, voc, op, transcript, reason="explicit vocative")
        if decided is not None:
            return decided

    # 4/5 — speech-act based routing.
    if op == "request_consult":
        actor = _actor_for_kind_or_id(engine, intent.target) or _first_actor_of_kind(engine, "consult")
        # Calling a consultant *is* summoning them — they answer the call.
        return Route("consult", op, actor_id=actor.id if actor else None,
                     present=True, summon=actor.id if actor else None,
                     reason="request_consult")

    if op == "ask_history" and intent.target:
        return _route_history_beat(engine, intent.target, op)

    if op in _STAFF_OPS:
        return Route("nurse", op, reason="staff speech-act")

    # confirm/cancel with nothing pending, or chitchat → bedside default is the
    # patient, unless the utterance opened by naming staff.
    if _addresses_staff(transcript):
        return Route("nurse", op, reason="addressed staff, no grounding")
    if _should_ask_back(intent, engine):
        return Route("head", op, ask_back=True, reason="ambiguous history ask")
    # Reasoned addressee (interpreter) — resolves a bare/undirected utterance so a
    # statement or thinking-aloud that names no one doesn't fall on the patient.
    reasoned = _route_by_addressee(engine, intent, op)
    if reasoned is not None:
        return reasoned
    # Genuinely unclear who a substantive request is for → the Head Physician asks the
    # resident to clarify, instead of silently defaulting the utterance onto the patient.
    if _is_unclear_request(intent, transcript):
        return Route("head", op, ask_back=True, reason="unclear addressee")
    return Route("patient", op, actor_id=_first_actor_of_kind(engine, "patient") and
                 _first_actor_of_kind(engine, "patient").id, reason="bedside default")


def _is_unclear_request(intent, transcript: str) -> bool:
    """A substantive utterance the interpreter couldn't attribute to anyone.

    Fires only when: the interpreter placed no addressee, the speech-act is bare
    conversation (orders/history already routed above), AND the line is a real
    request — a question or a full sentence — not a short ack/filler ("okay", "hmm").
    Keeps the Head Physician's clarify from nagging on thinking-aloud."""
    if getattr(intent, "addressee", None):
        return False
    if getattr(intent, "op", "chitchat") != "chitchat":
        return False
    text = (transcript or "").strip()
    return text.endswith("?") or len(text.split()) >= 5


def _route_by_addressee(engine, intent, op: str) -> Route | None:
    """Build a Route from the interpreter's reasoned ``addressee`` for a bare/undirected
    utterance. ``None`` when there's nothing usable, so the caller keeps the legacy
    bedside default. Honours presence for a named bystander."""
    addr = getattr(intent, "addressee", None)
    if not addr:
        return None
    if addr == "patient":
        actor = _first_actor_of_kind(engine, "patient")
        return Route("patient", op, actor_id=actor.id if actor else None,
                     reason="reasoned addressee: patient")
    if addr == "nurse":
        return Route("nurse", op, reason="reasoned addressee: nurse")
    actor = _first_actor_of_kind(engine, addr)   # ems / family / rt / consult
    if actor is None:
        return None
    if engine.is_present(actor.id):
        return Route(addr, op, actor_id=actor.id, present=True, reason="reasoned addressee")
    return Route(addr, op, actor_id=actor.id, present=False, reason="reasoned addressee (absent)")


# ── vocative / kind resolution ────────────────────────────────────────────────

# Interjections/fillers a resident may utter before naming who they're addressing
# ("Ah, paramedic…", "So, nurse…", "Okay, husband…"). We skip these so the vocative
# behind them still routes — dictation especially loves to prepend them.
_VOCATIVE_FILLERS = frozenset({
    "ah", "oh", "um", "uh", "er", "erm", "hmm", "so", "well", "okay", "ok", "alright",
    "right", "now", "hey", "hi", "hello", "yeah", "yep", "and", "but", "please",
})


def _leading_vocative(transcript: str):
    """The SpeakerKind named by the utterance's opening word, or None.

    Skips a short run of leading filler/interjection words so "Ah, paramedic, …"
    or "So, nurse, …" still resolve the person behind the filler. Stops at the
    first content word: if it isn't a vocative, there is no leading address."""
    words = re.findall(r"[a-z']+", (transcript or "").strip().lower())
    for w in words[:4]:
        if w in _VOCATIVE_FILLERS:
            continue
        return kind_for_word(w)   # first real word: a vocative, or None (no address)
    return None


def _route_to_kind(engine, kind, op: str, transcript: str, reason: str) -> Route | None:
    """Build a Route for an explicitly-named speaker kind, honouring presence."""
    kid = kind.id
    if kid in ("nurse", "patient"):
        return Route(kid, op, reason=reason)            # always in the room
    actor = _first_actor_of_kind(engine, kid)
    if actor is None:
        return None                                     # no such person in this case
    if engine.is_present(actor.id):
        return Route(kid, op, actor_id=actor.id, present=True, reason=reason)
    # Named but absent (and not a "call them" — that was handled upstream) → honest.
    return Route(kid, op, actor_id=actor.id, present=False, reason=reason + " (absent)")


def _route_history_beat(engine, beat_id: str, op: str) -> Route:
    """Route an ask_history to the person who holds the beat, honouring presence."""
    from backend.actor_voice import provider_speaker
    beat = next((h for h in engine.case.history if h.id == beat_id), None)
    kind = provider_speaker(beat.provider) if beat else "nurse"
    if kind in ("nurse", "patient"):
        return Route(kind, op, reason="history beat holder")
    actor = _first_actor_of_kind(engine, kind)
    if actor is None:
        return Route("nurse", op, reason="history beat holder (no actor)")
    if engine.is_present(actor.id):
        return Route(kind, op, actor_id=actor.id, present=True, reason="history beat holder")
    return Route(kind, op, actor_id=actor.id, present=False,
                 reason="history beat holder (absent)")


def _kind_of(engine, actor) -> str | None:
    if actor is None:
        return None
    from backend.actor_voice import actor_speaker
    return actor_speaker(actor)


def _first_actor_of_kind(engine, kind: str):
    from backend.actor_voice import actor_speaker
    return next((a for a in engine.case.actors if actor_speaker(a) == kind), None)


def _actor_for_kind_or_id(engine, token: str | None):
    """Resolve a chip/target token (an actor id, or a speaker-kind word) to a case actor."""
    if not token:
        return None
    direct = next((a for a in engine.case.actors if a.id == token), None)
    if direct is not None:
        return direct
    kind = kind_for_word(token)
    return _first_actor_of_kind(engine, kind.id) if kind else None


# ── pending / staff / ask-back (ported from director, presence-agnostic) ──────

def _resolves_pending(intent, transcript: str, engine) -> bool:
    pending = getattr(engine.run, "pending_action", None)
    if pending is None:
        return False
    if getattr(intent, "op", None) in ("confirm_pending", "cancel_pending"):
        return True
    from backend.nurse_brain import _is_affirmation, _is_cancellation, _med_dose
    if _is_affirmation(transcript) or _is_cancellation(transcript):
        return True
    if pending.get("status") == "awaiting_info" and pending.get("op") == "order_medication":
        return _med_dose(transcript, getattr(intent, "action_text", "")) is not None
    return False


def _summon_target(transcript: str, engine) -> str | None:
    """An absent roster actor the resident is asking to bring in ("call the husband",
    "page neurology"), or None. Returns the concrete actor id so the nurse can fetch
    them. Only fires for someone not already in the room."""
    text = (transcript or "").lower()
    if not _SUMMON_RE.search(text):
        return None
    for word in re.findall(r"[a-z']+", text):
        kind = kind_for_word(word)
        if kind is None or kind.id in ("nurse", "patient"):
            continue
        actor = _first_actor_of_kind(engine, kind.id)
        if actor is not None and not engine.is_present(actor.id):
            return actor.id
    return None


def _addresses_staff(transcript: str) -> bool:
    """Does the utterance open by naming a non-patient actor ('Nurse,'/'RT,'/…)?"""
    kind = _leading_vocative(transcript)
    return bool(kind and kind.id != "patient")


def _should_ask_back(intent, engine) -> bool:
    if not getattr(intent, "clarify", False):
        return False
    if getattr(intent, "op", None) != "ask_history":
        return False
    from backend.actor_voice import provider_speaker
    providers = {provider_speaker(h.provider) for h in engine.case.history}
    providers.discard("nurse")
    return len(providers) >= 2
