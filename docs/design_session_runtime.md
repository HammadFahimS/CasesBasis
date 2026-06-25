# Design Spec — Session Runtime (F1) + Push Channel (F2)

> Implements the two foundational gaps from [foundations.md](../foundations.md): the world must
> advance on a real clock even when the resident is silent, and the server must be able to *push*
> the resulting events to the room. This is the "always-running orchestrator with a push channel."

## Goals

1. **Autonomous clock.** Active, unpaused live sessions advance on a wall-clock tick — deterioration,
   gated results, cues, vitals/alarms fire *without* a resident utterance.
2. **Push channel.** The server streams engine events to the room as they happen.
3. **Non-breaking.** The existing per-turn REST pipeline keeps working; the runtime augments it.
4. **Deterministic core preserved.** The engine stays the single source of truth; the runtime only
   *drives* `engine.tick()` on a clock and *fans out* the `StepResult`s it already returns.

## Non-goals (later slices)

- Streaming TTS / audio over the socket, barge-in — Phase 2.
- The full "Director" emotional-state computation — this slice lands the *runtime*; the Director's
  routing/emotion logic moves in on top of it next.

---

## Architecture

```
                         ┌─────────────────────────────────────┐
   one asyncio task ───► │  SessionRuntime (backend/session_runtime.py)        │
   (app lifespan)        │  every TICK_SECONDS:                                 │
                         │    for each ACTIVE live session:                    │
                         │      results = advance_clock(session)  ← engine.tick │
                         │      events  = to_events(results, session)          │
                         │      bus.publish(session.id, events)   ────────────┐│
                         └─────────────────────────────────────┘             ││
                                                                              ▼▼
   browser ◄──── WebSocket /ws/voice/{session_id} ◄──── EventBus (per-session subscribers)
                         ▲
                         └─ REST turn (/api/voice/llm) also publishes its events to the same bus,
                            so the socket is the single ordered event stream the client renders.
```

- **`advance_clock(session)`** already exists (`voice_session.py`) and returns `list[StepResult]`. The
  runtime calls it on a timer instead of only inside a turn. **No engine change needed.**
- **EventBus**: in-process pub/sub keyed by `session_id` (an `asyncio.Queue` per subscriber). Bounded;
  drops to newest on overflow. Survives zero subscribers (events for a closed socket are no-ops).
- **Active set**: the runtime iterates the live `_SESSIONS` LRU, skipping `paused` and `ended` runs.
  Parked/evicted sessions stay frozen (existing contract) — they only tick once rehydrated by a turn.

### Threading / async model

The engine, DB, and `advance_clock` are **sync**; the runtime is **async**. The tick loop runs each
session's sync advance via `asyncio.to_thread` (or `run_in_executor`) so it never blocks the event
loop. SQLite is per-thread (`db.py` thread-local) → the executor thread gets its own connection; safe.
The existing sync REST handlers are unchanged (FastAPI already threadpools them).

### Event schema (the wire contract)

One JSON envelope per event, ordered, monotonically sequenced per session:

```json
{ "seq": 12, "t": 184, "kind": "state_change|result_ready|cue|vitals|speak|case_ended",
  "speaker": "nurse|patient|ems|family|rt|consult|system",
  "text": "…", "data": { … }, "ts": "2026-06-25T22:14:03Z" }
```

- `t` = engine elapsed seconds (so the client can place it on the case clock).
- `speaker` reuses the **cast registry** (`backend/actors.py`) — the client styles/voices by it.
- Derived from `StepResult` (+ the cue engine's `due_cues`, already surfaced in `nurse_brain._fire_cues`).
  The mapping `StepResult -> event` is pure and unit-testable offline.

### Idempotency / ordering

- Per-session monotonic `seq`; the client ignores out-of-order/duplicate seqs.
- The runtime and a concurrent REST turn both publish to the same bus; the engine's own mutation lock
  (single-threaded per session via the executor) keeps state transitions serialized. A turn and a tick
  never run the same session concurrently (a per-session `asyncio.Lock`).

### Lifecycle

- **Start:** `lifespan` startup launches the runtime task; shutdown cancels it cleanly.
- **Pause/Resume:** `pause` removes a session from the active set (clock frozen — matches today);
  `unpause` re-arms it. The socket stays open across pause.
- **Reconnect:** client sends its last-seen `seq`; server replays from a small per-session ring buffer
  (or says "resync" and the client re-fetches state via REST). Keeps the room consistent after a blip.

---

## Fallback / safety

- If the runtime task dies, **REST turns still advance the clock** (today's behavior) — degraded, not
  broken. The task is supervised and restarts.
- If the socket drops, the room still works turn-by-turn over REST; live events just pause until
  reconnect.
- Tick is cheap when idle (most sessions produce zero events most ticks); cost scales with *active*
  sessions, which is small for this app.

---

## Test plan (offline, no network)

- `to_events(StepResult…)` mapping — pure, table-driven.
- Runtime advances a fake session and publishes the expected events; paused/ended sessions don't tick.
- EventBus: publish/subscribe, overflow drop, multi-subscriber, unsubscribe on disconnect.
- Concurrency: a turn and a tick on the same session serialize (no double-fire of a transition).
- Engine untouched → existing engine/turn tests stay green.

---

## Rollout

1. Land the **runtime + EventBus + event mapping**, wired into `lifespan`, **feature-flagged**
   (`LIVE_RUNTIME=1`), publishing to the bus with **no client consumer yet** (observable via a debug
   log/endpoint). Zero behavior change when off.
2. Add the **WebSocket endpoint** `/ws/voice/{session_id}` (auth-scoped to the owning resident) and a
   thin client consumer that renders events into the existing Room log / monitor.
3. Flip REST turns to also publish to the bus, so the socket is the single ordered stream.
4. Only then build emotion/Director and streaming audio on top.
