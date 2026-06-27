"""In-memory live-session store for the voice-driven simulation room.

One :class:`VoiceSession` binds a :class:`SimulationEngine` to a DB-loaded
:class:`SimulatedCase` plus the spoken transcript so far. Sessions are keyed by
a generated id and scoped to the resident who started them.

In-memory only for now. The engine is serialisable (``snapshot``/``restore``),
so parking a run in the database can be layered on later without changing this
contract or the endpoints that consume it.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from backend.simulated_case_repo import get_case
from backend.simulation_engine import IntentMatcher, SimulationEngine, StepResult
from schemas.simulated_case import SimulatedCase


class VoiceSessionError(Exception):
    """Base class for session lookup/ownership failures."""


class CaseNotFound(VoiceSessionError):
    """Raised when a case id has no row, or a session id is unknown."""


class AccessDenied(VoiceSessionError):
    """Raised when a session is requested by a resident who doesn't own it."""


@dataclass
class VoiceSession:
    """A live voice-driven encounter."""

    id: str
    resident_id: str
    case: SimulatedCase
    engine: SimulationEngine
    history: list[dict] = field(default_factory=list)          # nurse channel
    patient_history: list[dict] = field(default_factory=list)  # patient channel
    cards: list[dict] = field(default_factory=list)            # diagnostics-drawer cards
    last_tick: float = field(default_factory=time.monotonic)
    # Live status, mirrored from the parked DB row so the Session Runtime can tell
    # which sessions to advance. A paused run stays in the live cache (so resume is
    # instant) but must NOT tick — its clock would deteriorate a parked patient.
    status: str = "active"                                     # active | paused | ended
    # Serialises a runtime tick against a concurrent REST turn on the same session,
    # so a clock advance never interleaves with a turn's engine mutation. A plain
    # threading.Lock works across both FastAPI's sync threadpool and the runtime's
    # executor thread. Excluded from equality (it's transient, never serialised).
    lock: threading.Lock = field(default_factory=threading.Lock, compare=False, repr=False)


# Live-session LRU. Bounded because every session is also parked to the DB and
# rehydrates on demand (get_session), so evicting a cold entry is lossless — it
# just costs one reload on the next touch. Keeps memory flat under long uptime.
_SESSIONS: "OrderedDict[str, VoiceSession]" = OrderedDict()
_MAX_LIVE_SESSIONS = 256


def _remember(session: VoiceSession) -> None:
    """Cache a session as most-recently-used, evicting the coldest past the cap."""
    _SESSIONS[session.id] = session
    _SESSIONS.move_to_end(session.id)
    while len(_SESSIONS) > _MAX_LIVE_SESSIONS:
        _SESSIONS.popitem(last=False)   # drop least-recently-used (DB still has it)


def _intent_matcher() -> IntentMatcher:
    """Build a fresh LLM-backed action matcher (lazy import avoids a cycle)."""
    from backend.nurse_brain import make_intent_matcher
    return make_intent_matcher()


def start_session(
    case_id: str, resident_id: str, intent_matcher: IntentMatcher | None = None
) -> VoiceSession:
    """Load a case from the DB and spin up a fresh engine-backed session."""
    row = get_case(case_id)
    if row is None:
        raise CaseNotFound(case_id)
    case = SimulatedCase.model_validate(row["case"])
    engine = SimulationEngine(case, intent_matcher=intent_matcher)
    session = VoiceSession(
        id=secrets.token_urlsafe(12),
        resident_id=resident_id,
        case=case,
        engine=engine,
    )
    _remember(session)
    persist_session(session)  # park it from the first moment so it survives a restart
    return session


def _rehydrate(row: dict) -> VoiceSession:
    """Rebuild a live session from a parked DB row (engine restored, clock frozen)."""
    case_row = get_case(row["case_id"])
    if case_row is None:
        raise CaseNotFound(row["case_id"])
    case = SimulatedCase.model_validate(case_row["case"])
    engine = SimulationEngine(case, intent_matcher=_intent_matcher())
    engine.restore(row["snapshot"])
    session = VoiceSession(
        id=row["id"],
        resident_id=row["resident_id"],
        case=case,
        engine=engine,
        history=row.get("history") or [],
        patient_history=row.get("patient_history") or [],
        cards=row.get("cards") or [],
        # Reset the tick to now so wall-time spent parked does NOT advance the
        # engine clock — a parked patient must not deteriorate. Pause freezes time.
        last_tick=time.monotonic(),
    )
    _remember(session)
    return session


def get_session(session_id: str, resident_id: str) -> VoiceSession:
    """Return a session (live cache, else rehydrated from DB) if owned by this resident."""
    session = _SESSIONS.get(session_id)
    if session is None:
        from backend import voice_persistence
        row = voice_persistence.load(session_id)
        if row is None:
            raise CaseNotFound(session_id)
        if row["resident_id"] != resident_id:
            raise AccessDenied(session_id)
        session = _rehydrate(row)
    else:
        _SESSIONS.move_to_end(session_id)   # mark as recently used
    if session.resident_id != resident_id:
        raise AccessDenied(session_id)
    return session


def persist_session(session: VoiceSession, status: str = "active") -> None:
    """Write the session's current state through to durable storage (best-effort).

    Also mirrors ``status`` onto the live object so the Session Runtime sees
    pause/resume/end without a DB read — pause parks as ``paused`` and must not tick.
    """
    session.status = status
    from backend import voice_persistence
    voice_persistence.save(session, status)


def end_session(session_id: str) -> None:
    """Conclude a session: mark it ended in storage and drop the live copy."""
    from backend import voice_persistence
    session = _SESSIONS.get(session_id)
    if session is not None:
        voice_persistence.save(session, status="ended")
    else:
        voice_persistence.mark_ended(session_id)
    _SESSIONS.pop(session_id, None)


def advance_clock(session: VoiceSession) -> list[StepResult]:
    """Advance the engine clock by the real time elapsed since the last turn.

    Time/inaction transitions therefore fire while the resident deliberates —
    the world keeps moving between utterances. Returns any state changes."""
    now = time.monotonic()
    delta = int(now - session.last_tick)
    session.last_tick = now
    # A concluded case is frozen: don't let wall-time inflate the final elapsed.
    if delta <= 0 or session.engine.run.ended:
        return []
    return session.engine.tick(delta)
