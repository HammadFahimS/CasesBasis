# Room of Agents — the live-sim routing rewrite

> Status: **design, approved for build** (2026-06-27). Supersedes the routing core
> described in `multi-agent-plan.md` §2 and pulls Phase 5 (CharacterAgents) forward
> as the *actual* fix. Phases 2–3 (server-side Director, `pending_action`) were
> correct but were patches on a topology whose root defect this document removes.

## 0 · The defect, stated once

There is **no model of the room**. There is one nurse pipeline (`nurse_brain`)
with bystanders bolted on as conditional detours inside `director.voice_reply`,
and **the nurse is the universal fallback** — an LLM that always produces a
confident, reasonable-sounding line. So every routing or grounding failure is
*laundered into nurse filler instead of surfacing.* Failures are invisible; that
is why, to the resident, "nothing happens."

Evidence (session `brw6ct06CEYorny-`, a real playtest):

- Resident addressed the **husband** four times (turns 25/27/29/31) before his
  beats unlocked at 300 s. Each time the **nurse** answered "no new information at
  this time." The husband could not speak; the engine returned *"not available
  yet"*, `result.revealed` went `None`, and the turn fell to the nurse sink.
- "Husband, what do you know?" (turn 52) → nurse "I'm here, Resident. Patient is
  stable…". The husband was addressed; the nurse covered for him.
- ECG read aloud (turn 13) but CXR "it's on the screen" (turn 15) — no policy
  boundary on diagnostics, pure LLM phrasing variance.
- The opening brief plays sometimes and not others — it races WS connect / audio
  unlock and is not tied to a deterministic "session ready" event.

### Three structural facts that cause all of it

1. **Presence is not a routing input.** `ActorRole.available_after_seconds`
   exists (husband = 300 s) but the Director never reads it. Addressing someone
   absent has no honest answer — it decays into a gated-beat miss → nurse filler.
2. **Bystanders have no positive routing.** In `director.voice_reply` a bystander
   speaks only as a *side effect* of a successful `ask_history` reveal
   (`op == ask_history AND result.revealed AND provider != nurse`). There is no
   "address the husband → the husband answers." A `chitchat` classification, a
   wrong beat target, or an already-revealed beat all make the husband mute and
   the nurse speak for him.
3. **Addressee and content are one entangled guess.** A single flat `NurseIntent`
   (op + target) carries both *who* and *what*. A small interpreter miss
   mis-routes the whole turn — into the sink.

## 1 · The four contracts (the actual fix)

The graph shape is router-node → actor-leaf-nodes over a shared blackboard (the
engine = single source of clinical truth; each actor reads only its slice). But
the leverage is **not** a graph library — it is these four contracts. A graph
without them has the same bug with more boxes.

### C1 · Presence is first-class and gates routing
The engine owns `present_actors: set[str]` (actor ids), derived from
`available_after_seconds` plus explicit summon events ("call the husband"). The
room *knows* who is here. Addressing an absent actor yields a **deterministic,
honest** line ("The husband isn't here yet — want me to call him?"), spoken by the
nurse/system as a *router* message — never silence, never invented content, never
a beat-miss laundered as filler.

### C2 · The Router returns an explicit Addressee on two orthogonal axes
Per utterance the Router decides **`(addressee, speech_act)`**:

- `addressee` ∈ the *full* roster incl. consultants and the patient, resolved by,
  in priority: **forced chip** → **explicit vocative** (full roster) → **active-
  conversation continuity** (we're mid-dialogue with the husband; "tell me more"
  stays with him) → **content/intent** → honest ask-back.
- `speech_act` ∈ `{ask_history, order_med, order_test, exam, request_consult,
  diagnose, confirm, cancel, smalltalk}`.

These are independent. "Husband, what meds is she on?" = `(husband, ask_history)`
cleanly, not a beat-target lottery. Staff orders are `(nurse, order_*)`. The
patient is just one possible addressee, not a fallback default.

### C3 · Every actor is an Agent with a bounded source and an in-character refusal
An actor answers **only** from the knowledge it holds (its history beats, its
role). Asked something it cannot know, it declines *as itself* ("I'd have to ask
her doctor") — it never invents, and **no agent may answer for another.** The
nurse becomes one agent among peers; it is **removed as the universal sink.** When
nothing can answer, the *Router* says so honestly — the nurse does not fabricate a
cover.

### C4 · Hard diagnostics boundary
Tests and exams produce a **card** (the visual artifact) plus a terse availability
announcement from the acting actor ("ECG's up on the monitor"). The actor **never
reads the interpretation aloud.** Deterministic policy in the dispatch/voice seam,
not an LLM hope — this permanently fixes the ECG/CXR inconsistency.

## 2 · Target data model

```
SimulationEngine / RunState
  present_actors: set[str]            # C1 — who is in the room now
  active_addressee: str | None        # C2 — who we're mid-conversation with
  pending_action: dict | None         # (kept from Phase 3)

backend/routing.py  (new — the Router)
  @dataclass Route:
      addressee: str                  # actor id / speaker kind ('nurse','husband',…)
      speech_act: str                 # ask_history | order_* | exam | … | smalltalk
      target: str | None              # beat id / investigation id / med, when known
      reason: str                     # why this route fired (for trace + tests)
      absent: bool = False            # addressee resolved but not present (C1)
  def route(utterance, engine, history, forced_addressee) -> Route

backend/agents.py  (new — the actor agents)
  class Agent(Protocol):
      def can_answer(route, engine) -> bool
      def respond(route, engine, session, utterance) -> AgentReply   # (text, speaker, card?)
  registry keyed off actors.SPEAKERS + the case roster
  NurseAgent, PatientAgent, BystanderAgent (ems/family), RTAgent, ConsultAgent
  — each bounded to its source; refusal is in-character, never fabrication.
```

The **engine stays the single source of clinical truth.** `routing.py` and
`agents.py` only decide *who speaks* and *let them read their slice*; they never
invent a clinical value. `dispatch` (deterministic engine ops) is reused as-is —
the rewrite is above it.

## 3 · What is reused vs replaced

| Keep (reuse) | Replace |
|---|---|
| `simulation_engine` ops, `dispatch`, `pending_action` protocol | `director.resolve_addressee` priority chain → `routing.route` |
| `actors.SPEAKERS` registry, voices, `kind_for_word` | `director.voice_reply`'s bystander side-effect detour → `agents` |
| `actor_voice` / `patient_voice` / `consult_voice` `_voice_as` phrasing | the nurse-as-universal-fallback in `_resolve_turn` |
| the headless harness + `pending_action` | flat single-axis `NurseIntent` routing semantics |
| cue engine (`due_cues`) | undisciplined diagnostics phrasing (→ C4 boundary) |

## 4 · Build phases (each lands behind the harness, old path as fallback to parity)

- **RA-1 · Presence (C1). ✅ DONE.** `is_present`/`present_actor_ids`/`summon`/
  `arrival_wait` on the engine; `run.summoned_actors` (snapshot-additive, legacy →
  `[]`). Bedside team present at t=0; husband absent until 300s or summon;
  consultants never in-room by the clock. A history beat's timer now defers to its
  holder's presence (an early-summoned husband can answer). Tests:
  `tests/test_simulated_case.py` (presence block).
- **RA-2 · Router (C2). ✅ DONE.** `backend/routing.py` `route() → Route(addressee,
  speech_act, actor_id, present, summon, ask_back, reason)`. `run_turn` branches on
  it: ask-back / honest-absent (`_absent_actor_turn`) / summon (`_summon_turn`) /
  patient / present-bystander (positive beat routing) / staff. The old
  `director.resolve_addressee` chain reached parity and was **removed** (director now
  owns only cue firing, voice routing, ask-back phrasing). Tests:
  `tests/test_routing.py` + RA guards in `tests/test_live_sim_conversations.py`.
- **RA-3 · Agents (C3). ✅ DONE.** `backend/agents.py` makes the contract explicit:
  `ROOM_AGENTS` bounds each kind to its own knowledge, and `refuse()` gives a present
  actor an in-character decline when asked outside what they hold. `run_turn` now
  force-routes a history/chitchat turn to the addressed present bystander's beat, or
  — if they hold none (e.g. the RT has no history) — to `_refusal_turn` voiced by
  *them*, never the nurse. This closes the last nurse-as-sink residue. Orders to an
  actor still dispatch normally. Test:
  `test_present_actor_refuses_in_character_when_asked_outside_their_knowledge`.
  (Full CharacterAgents — constitution + EmotionalState + voice binding per actor —
  remain `multi-agent-plan.md` Phase 5; `agents.py` is their seam.)
- **RA-4 · Diagnostics boundary + brief (C4 + folded side issues). ✅ DONE.**
  Visual studies (`ecg`/`imaging`) now get a deterministic availability announce
  (`director._announce_study`) — never routed through the embellishing LLM, so the
  interpretation can't be read aloud; it stays card-only. Labs/bedside still spoken.
  Tests: `test_visual_study_*` / `test_lab_result_is_still_spoken`. The "nurse
  doesn't brief on load" race was a **client** bug: `voice_room.html playHandoff`
  set `introPlayed = true` *before* TTS, so a transient TTS failure swallowed the
  one-time briefing forever — now flagged only after it actually plays (headless-
  untestable; verify by playtest). Server `opening_narration` already falls back
  deterministically.
- **RA-5 · Cutover & cleanup. ✅ DONE.** Dead `resolve_addressee`/`RouteDecision`
  chain removed from `director.py` (now: cue firing, voice routing, ask-back phrasing,
  diagnostics announce). One Router (`backend/routing.py`), one presence model
  (`simulation_engine`), one agent seam (`backend/agents.py`). Suite: 503 passed.

- **RA-6 · Presence-gated cues + arrivals + the room talks back. ✅ DONE.**
  Second playtest (session `3C8HDFREz_-8Hk6I`) surfaced three more defects, all now
  fixed and guarded:
  1. **Cues fired before the actor they reference was present.** The husband cues
     ("ask him about her symptoms", "he's arrived and is waiting") fired on a
     state-linger timer (`due_cues` at 75s in-state) with no presence awareness, so
     the room announced his arrival while the honest-absent path was still saying "1
     minute out" — a flat contradiction. Fix: `Cue.requires_present` (a presence gate
     in the schema) + `due_cues` skips a cue until every referenced actor `is_present`
     (and does *not* mark it fired, so it becomes eligible on arrival). The two
     husband cues now carry `requires_present: ["husband"]`.
  2. **Arrival was never announced at the right moment.** Added `RunState
     .announced_arrivals` + `engine.due_arrivals()` — the nurse announces "her
     husband's just arrived" the instant presence flips true (clock hits 300s),
     exactly once; a `summon` suppresses the duplicate clock announce. Fired through
     `director.fire_cues` so it reaches both the turn payload and the live WS.
  3. **Tasks given to actors never reported back (the open loop).** "RT, get me a VC"
     was even mis-classified as `request_consult(rt)` and never reached the engine
     (VC/NIF — the numbers that decide intubation — never came back). Two fixes:
     the RT is excluded from `consult_voice.consultants` (it's a bedside team member
     holding orderable tests, not a phone consult); and a **deferred-result subsystem**
     — `InvestigationDef.result_latency_seconds` + `result_actor_id`, a
     `RunState.deferred_events` queue, `engine.order_investigation` schedules a
     callback instead of a dead "pending", and `engine.due_deferred()` drains it.
     `director.fire_deferred` voices the holder's terse, signal-only report (C4
     discipline) and drops the card in the panel. Plumbed through both the turn POST
     (`payload["deferred"]`) and the live WS (`kind="result"` carrying the card).
     This is the generic "an actor does something after N seconds → reports back +
     updates diagnostics" loop. Tests: `test_simulated_case.py` (arrivals, cue
     gating, deferred scheduling/firing/roundtrip) + `test_live_sim_conversations.py`
     (RT-not-a-consultant, bedside reports back with card, nurse announces arrival).
     Suite: 517 passed.

---

**All four contracts are in place.** C1 presence-gated routing, C2 the two-axis
Router, C3 bounded agents with in-character refusal, C4 the diagnostics boundary.
Two browser changes (the handoff-replay fix and the order-announce phrasing) are
headless-untestable and need a playtest. The remaining future work is
`multi-agent-plan.md` Phase 5 (full CharacterAgents + EmotionalState, hung off
`backend/agents.py`) and Phase 6 (ElevenLabs/latency) — separate from this rewrite.

## 5 · Test strategy (the rewrite is only as good as this)

Today the harness asserts `speaker == nurse`. Under this model it asserts the
**Route** and the **reason**, so a routing regression *fails a test* instead of
silently sounding plausible. Every reported symptom becomes a permanent guard:

- `addressing an absent actor is honest, not nurse filler` (the husband-before-300s bug)
- `addressing a present bystander reaches that bystander` (the 6-7-tries bug)
- `diagnostics announce availability, never read interpretation` (ECG/CXR)
- `the opening brief is emitted exactly once on session-ready` (the missing brief)
- existing S1/S2/S3 guards continue to pass throughout.

## 6 · Non-goals / guardrails (unchanged)

- The engine remains the only source of clinical truth — no agent invents a value.
- Additive-first snapshots; in-flight runs must not strand (same rule as Phase 4).
- Our own Router, not ElevenAgents (see `multi-agent-plan.md` §1.1) — we need the
  deterministic, headless-testable seam this whole document depends on.
