"""Session Runtime — the autonomous clock + event bus for the live voice sim.

Foundational change (F1 + F2 in ``foundations.md``). Today the case clock only
advances inside a resident turn, so a deteriorating patient *freezes* while the
resident deliberates. This module runs a background loop that advances every
active live session on a wall-clock and publishes the resulting events to a
per-session bus, which a WebSocket (added next) streams to the room.

Design constraints honoured here:

* **The engine stays the single source of truth.** We only *drive* ``engine.tick``
  (via :func:`advance_clock`) on a timer and *fan out* the ``StepResult``s it
  already returns — plus any cues that came due. No clinical content is invented.
* **Deterministic & offline-testable.** :func:`to_events` is a pure mapping; the
  loop and bus are exercised without network or real time in the tests.
* **Non-breaking & flagged.** :func:`start_runtime` is a no-op unless ``LIVE_RUNTIME``
  is truthy. With it off, behaviour is exactly as before (REST turns still advance
  the clock).
* **Serialised per session.** Each session carries a ``lock``; a tick and a REST
  turn never mutate the same engine concurrently.

This module deliberately has *no* WebSocket code — the bus is the seam the socket
layer will subscribe to next.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from backend.simulation_engine import StepResult

logger = logging.getLogger(__name__)

TICK_SECONDS = float(os.getenv("LIVE_RUNTIME_TICK_SECONDS", "2"))
_RING = 64  # per-session replay buffer for socket reconnects (slice 2)


def runtime_enabled() -> bool:
    return os.getenv("LIVE_RUNTIME", "").strip().lower() in {"1", "true", "yes", "on"}


# ── Events ───────────────────────────────────────────────────────────────────

@dataclass
class SessionEvent:
    """One thing that happened in a session, ready to send to the room.

    ``seq`` is stamped by the bus (monotonic per session); ``t`` is the engine
    clock in seconds so the client can place it on the case timeline.
    """

    kind: str                      # state_change | condition | cue | case_ended
    t: int
    speaker: str = "system"        # cast tag: nurse|patient|ems|family|rt|consult|system
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    seq: int = 0
    ts: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq, "t": self.t, "kind": self.kind,
            "speaker": self.speaker, "text": self.text, "data": self.data, "ts": self.ts,
        }


def to_events(
    clock: list[StepResult], cues: list[dict], elapsed: int,
    vitals: dict | None = None,
) -> list[SessionEvent]:
    """Pure mapping: engine outputs for one tick → ordered room events.

    A transition becomes a ``state_change``; a non-transition message (a condition
    change with no state move) becomes a ``condition``; cues carry their actor's
    voice. A run that just ended is closed with a ``case_ended`` event. ``vitals``
    (the engine's live vitals after the tick) ride on the engine-driven events so
    the room's monitor reacts to deterioration without a separate fetch.
    """
    events: list[SessionEvent] = []
    for r in clock:
        if r.transitioned:
            events.append(SessionEvent(
                kind="state_change", t=elapsed, speaker="system",
                text=r.message or f"Entered {r.state_name}",
                data={"from_state": r.from_state, "state_id": r.state_id,
                      "state_name": r.state_name, "critical": r.critical_error,
                      "vitals": vitals},
            ))
        elif r.message:
            events.append(SessionEvent(
                kind="condition", t=elapsed, speaker="system",
                text=r.message, data={"critical": r.critical_error, "vitals": vitals},
            ))
        # A terminal transition sets ended on the run; surface it once, after the move.
    for cue in cues:
        events.append(SessionEvent(
            kind="cue", t=elapsed,
            speaker=cue.get("speaker", "nurse"), text=cue.get("text", ""),
        ))
    return events


# ── Event bus ────────────────────────────────────────────────────────────────

class EventBus:
    """In-process per-session pub/sub. Stamps a monotonic ``seq`` and keeps a small
    ring buffer so a reconnecting socket can replay what it missed (slice 2).

    Bounded queues drop the *oldest* event on overflow — a slow/absent consumer
    never blocks the runtime or grows memory without bound."""

    def __init__(self, maxsize: int = 256) -> None:
        self._subs: dict[str, set[asyncio.Queue]] = {}
        self._seq: dict[str, int] = {}
        self._ring: dict[str, deque[dict]] = {}
        self._maxsize = maxsize

    def publish(self, session_id: str, events: list[SessionEvent]) -> list[dict]:
        """Stamp ``seq``/``ts`` and deliver to subscribers + the ring. Returns the
        stamped dicts (handy for tests and for the REST path to echo)."""
        out: list[dict] = []
        ring = self._ring.setdefault(session_id, deque(maxlen=_RING))
        for ev in events:
            self._seq[session_id] = self._seq.get(session_id, 0) + 1
            ev.seq = self._seq[session_id]
            ev.ts = datetime.now(timezone.utc).isoformat()
            payload = ev.to_dict()
            out.append(payload)
            ring.append(payload)
            for q in list(self._subs.get(session_id, ())):
                if q.full():
                    try:
                        q.get_nowait()  # drop oldest
                    except asyncio.QueueEmpty:
                        pass
                q.put_nowait(payload)
        return out

    def subscribe(self, session_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._maxsize)
        self._subs.setdefault(session_id, set()).add(q)
        return q

    def unsubscribe(self, session_id: str, q: asyncio.Queue) -> None:
        subs = self._subs.get(session_id)
        if subs:
            subs.discard(q)
            if not subs:
                self._subs.pop(session_id, None)

    def replay(self, session_id: str, after_seq: int) -> list[dict]:
        return [e for e in self._ring.get(session_id, ()) if e["seq"] > after_seq]


# ── Runtime ──────────────────────────────────────────────────────────────────

class SessionRuntime:
    """Background loop that advances active live sessions and publishes their events."""

    def __init__(self, bus: EventBus, tick_seconds: float = TICK_SECONDS) -> None:
        self.bus = bus
        self.tick_seconds = tick_seconds
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="session-runtime")
            logger.info("SessionRuntime started (tick=%ss)", self.tick_seconds)

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            logger.info("SessionRuntime stopped")

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self.tick_seconds)
            try:
                await self.tick_once()
            except Exception:  # the loop must survive a bad session
                logger.exception("SessionRuntime tick failed")

    async def tick_once(self) -> None:
        """Advance every currently-live session once. Each session's sync work runs
        in a worker thread so the event loop is never blocked."""
        from backend.voice_session import _SESSIONS
        for session in list(_SESSIONS.values()):
            try:
                events = await asyncio.to_thread(self._tick_session, session)
            except Exception:
                logger.exception("tick failed for session %s", session.id)
                continue
            if events:
                self.bus.publish(session.id, events)

    @staticmethod
    def _tick_session(session: Any) -> list[SessionEvent]:
        """Sync, lock-guarded: advance one session's clock and collect its events.
        Returns ``[]`` for paused/ended sessions or an uneventful tick."""
        from backend.nurse_brain import _fire_cues
        from backend.voice_session import advance_clock, persist_session
        with session.lock:
            if session.status != "active" or session.engine.run.ended:
                return []
            clock = advance_clock(session)
            cues = _fire_cues(session)
            if not clock and not cues:
                return []
            vitals = session.engine.current_vitals().model_dump(exclude_none=True)
            events = to_events(clock, cues, session.engine.run.elapsed_seconds, vitals)
            if session.engine.run.ended:
                events.append(SessionEvent(
                    kind="case_ended", t=session.engine.run.elapsed_seconds,
                    speaker="system", text="The case has ended.",
                ))
            # Park the advance so a restart doesn't lose pushed deterioration/cues.
            persist_session(session, "ended" if session.engine.run.ended else "active")
            return events


# ── Process-wide singletons (started/stopped by the app lifespan) ─────────────

_BUS = EventBus()
_RUNTIME: SessionRuntime | None = None


def get_bus() -> EventBus:
    return _BUS


def get_runtime() -> SessionRuntime | None:
    return _RUNTIME


async def start_runtime() -> None:
    """Launch the runtime iff ``LIVE_RUNTIME`` is set. Safe to call once at startup."""
    global _RUNTIME
    if not runtime_enabled():
        logger.info("SessionRuntime disabled (set LIVE_RUNTIME=1 to enable)")
        return
    _RUNTIME = SessionRuntime(_BUS)
    await _RUNTIME.start()


async def stop_runtime() -> None:
    global _RUNTIME
    if _RUNTIME is not None:
        await _RUNTIME.stop()
        _RUNTIME = None
