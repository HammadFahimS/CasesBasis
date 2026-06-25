# CasesBasis — Foundations to Strengthen Before the Feature Push

> Before building the [live-sim vision](live_sim_vision.md), this is the structural work that keeps
> the app **maintainable and production-viable** as agents, voices, and cases multiply. Findings are
> grounded in the current code (June 2026), with severity and a recommended sequence.
>
> **What's already solid (don't touch):** the SQLite layer is well-engineered — WAL mode,
> `BEGIN IMMEDIATE` write transactions, per-thread connections, `busy_timeout`, **and a real
> file-based migration runner** (`backend/db.py::run_migrations` + `migrations/*.sql` +
> `schema_migrations` table). Persistence is clean JSON-blob snapshots. Build on this; don't rebuild it.

---

## The two foundational gaps (do these first — everything else depends on them)

### F1 · No autonomous clock — the world only ticks when the resident speaks 🔴
**Evidence:** the only time advance is `voice_session.advance_clock`, called inside the per-turn
pipeline (`nurse_brain.run_turn`). There is **no background loop** anywhere (`grep` for
schedulers/threads finds only the DB thread-local, a FAISS loader, and a PTB-XL lock). The JS timer is
display-only.

**Why it's the #1 issue:** the entire live-sim premise is a patient on a real clock — deterioration,
**gated results arriving** (VC/NIF 60s, VBG 120s, husband 300s), cues, alarms. Today **none of that
fires while the resident is silent.** If they stop talking, time freezes. Every realism feature in the
vision (audible deterioration arc, "she crashed while you were thinking," monitor alarms, proactive
cues) is blocked on this. This is exactly the *"orchestrator that keeps running and ensures time-bound
events"* you intuited.

**The fix — a Session Runtime (the Director's home):** an async tick loop that, on an interval (~1s),
advances every *active, unpaused* live session, runs the engine, and **emits the resulting events**
(state change, result ready, cue, vitals/alarm, deterioration). Bounded to live sessions (parked ones
stay frozen — already the contract). Pure-engine ticking stays deterministic; the runtime just drives
it on a wall-clock and fans out events. This becomes the home for the thin **Director** from the
vision doc (routing + cues + emotional-state computation).

### F2 · No realtime transport — everything is POST request/response 🔴
**Evidence:** all `/api/voice/*` endpoints are request/response; **no WebSocket or SSE anywhere**
(`grep` confirms). The server cannot *push* anything to the room — it can only answer when asked.

**Why it matters:** F1 is useless without a channel to deliver its events. You can't push "result is
back," "patient deteriorating," "monitor alarming," stream audio/captions, or support **barge-in**
over plain POST. This is the transport half of the same problem.

**The fix:** add **one WebSocket** per live session (`/ws/voice/{session_id}`) as the event + audio
channel; keep REST for stateless ops (start/resume/list/delete). The Session Runtime (F1) publishes
engine events to the socket; the client renders them (log lines, alarms, captions, spoken audio).
Pairs naturally with streaming TTS/STT later.

> **F1 + F2 together = the "always-running orchestrator with a push channel."** This is the single
> most important base change. Land it before any voice-realism feature, because they all ride on it.

---

## Maintainability (do alongside the first features, not after)

### F3 · `web_app.py` is a 5,467-line monolith 🟠
**Evidence:** `HTML_PAGE` starts at line 1721 — so **~3,700 lines of inline HTML/JS/CSS** live inside a
Python string, plus ~1,700 lines of route handlers in one file. The home/analytics pages were already
extracted to `backend/static/*.html`; the **workbench/simulator frontend never was.**

**Why it matters:** this is the #1 thing that becomes "unmanageable" as the room grows (presence
indicators, captions, monitor, ambient, transitions). Editing UI inside a Python string has no
tooling, no component boundaries, and bloats diffs.

**The fix (incremental, low-risk):**
1. Finish the extraction already in motion: move the workbench `HTML_PAGE` into
   `backend/static/voice_room.html` / a workbench template, served via the existing `<!--CHROME-->` /
   `_inject_drawer` mechanism. No behavior change.
2. Split routes into **`APIRouter` modules** by domain (`routers/voice.py`, `routers/cases.py`,
   `routers/dashboard.py`, `routers/auth.py`) and `include_router` them. `web_app.py` shrinks to app
   wiring + lifespan.

### F4 · The runtime database is committed to git 🟠
**Evidence:** `casesbasis.db` (3.1 MB) is tracked; only `*.db-wal`/`*.db-shm` are gitignored. It shows
as **modified on every run** (it's dirty in the current branch).

**Why it matters:** binary churn in history, merge conflicts, and it conflates **seed data** (gold
cases, which already live in `cases/*.json` and seed on startup) with **runtime data** (sessions,
evals). Not production-viable.

**The fix:** gitignore `casesbasis.db`; rely on `seed_gold_cases()` (already runs at startup) +
migrations to build a fresh DB. Ship **`scripts/reseed_case.py <case_id>`** so gold-case JSON edits
reach the DB without the manual `UPDATE` footgun *(see memory `gold-case-db-not-file`)*. Provide a
one-time `git rm --cached casesbasis.db`.

### F5 · No unified agent runtime — provider plumbing is copy-pasted per module 🟠
**Evidence:** `voice_service`, `actor_voice`, `patient_voice`, `voice_debrief` each define their own
`_complete` seam and repeat `build_llm_client / get_llm_provider / resolve_model`. Interactive
timeouts are long (httpx 60s; `agents/base` urllib 120s) — a slow provider stalls a turn for up to a
minute with no retry/observability.

**Why it matters:** the vision multiplies agents (constitution + per-case flavor + director +
emotional voicing). Without a shared runtime, every new agent re-implements prompt assembly, timeout,
retry, fallback, and we get N subtly-different LLM call sites — the "unmanageable agents" risk you
flagged.

**The fix:** one small **`AgentRuntime`** abstraction: takes a system+user prompt (or a structured
schema), applies **interactive-grade timeouts (~10–15s) + one retry + soft fallback**, and centralizes
tracing/metrics. All voice/character agents call it. The **constitution registry** (base per-kind
prompt) plugs in here, with case flavor layered on top — directly enabling the vision's agent model.

---

## Hardening (schedule deliberately, not urgent)

### F6 · Engine snapshot versioning 🟡
`engine.snapshot()` / `restore()` is unversioned JSON. When the engine's internal shape changes,
**parked sessions break on resume.** Add a `snapshot_version` and a tiny upgrade path so a deploy
doesn't strand in-flight runs.

### F7 · Case-schema migration/backfill path 🟡
The vision adds fields to the case schema (`voice_id`, `voice_settings`, per-state emotional arcs on
`ActorRole`). The case is stored as one `case_json` blob — old rows won't have the new fields. Add a
**case-migration step** (reuse the migrations runner concept, but for `case_json` content) + the
reseed script (F4), so schema growth doesn't silently break existing cases. Keep Pydantic defaults
generous so old blobs still validate.

### F8 · Interactive timeout/latency budget 🟡
60–120s timeouts are wrong for a live turn. Define a **latency budget** (e.g. <1.5s to first audio),
set short timeouts with fallback in the `AgentRuntime` (F5), and add per-stage timing telemetry
(STT → interpret → dispatch → speak → TTS) to diagnostics. The serial turn chain also wants to go
**async/streamed** once F2 lands.

---

## Recommended sequence

```
1. F4  gitignore the DB + reseed script        (1 small PR, unblocks clean dev, low risk)
2. F1  Session Runtime / tick loop             (the core architectural change)
3. F2  WebSocket event+audio channel           (delivers F1's events; do with F1)
4. F5  AgentRuntime + constitution registry    (before multiplying agents)
5. F3  extract workbench frontend + routers     (incremental, parallelizable)
6. F6/F7/F8  hardening as the schema/agents grow
```

**Why this order:** F4 is a quick win that cleans the repo. F1+F2 are the load-bearing change every
realism feature needs — do them as a pair. F5 must precede the agent-heavy vision work or we entrench
copy-paste. F3 can proceed in parallel and de-risks all future UI work. F6–F8 harden as we go.

> **The throughline:** keep the **engine as deterministic truth**, add an **always-running runtime +
> push channel** around it, route all model calls through **one agent runtime**, and split the
> monolith. That base supports the full vision without the code or the agent count becoming
> unmanageable — and it generalizes to diverse cases because none of it is myasthenic-crisis-specific.
