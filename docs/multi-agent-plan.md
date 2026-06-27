# CasesBasis Live Sim — Multi-Agent Routing Rebuild & Test Harness

> Plan to fix the structural defects at the heart of the live voice sim and turn it into a
> robust multi-agent room. Companion to [live_sim_vision.md](../live_sim_vision.md) (the product
> north star) and [elevenlabs_capabilities.md](../elevenlabs_capabilities.md) (the voice supplier).
> This doc is the **how**: the concrete phases, plus a headless conversation test harness so the
> system can be validated without anyone driving the browser.

---

## 0 · The one-sentence problem

**Turn-spanning state lives in the transcript, not in the engine** — so the LLM fakes continuity
by re-reading history, and the moment a turn doesn't restate everything, the room breaks. Every bug
below is a symptom of that single pathology.

### Three confirmed symptoms

| # | Symptom | Root cause | Evidence |
|---|---|---|---|
| S1 | Address EMS/husband, the **patient** answers | Routing is decided **client-side** in `detectAddressee()` (keyword regex, `else → patient` default), then split across two endpoints; the server **Director never decides who speaks** | [voice_room.html:2591](../backend/static/voice_room.html#L2591), [routers/voice.py:383](../backend/routers/voice.py#L383) |
| S2 | Transcript tags **every** voice "nurse" | Replies persisted with hardcoded `role:"nurse"`; the real `speaker` is computed, sent to the client, never written | [nurse_brain.py:520](../backend/nurse_brain.py#L520), [director.py:109](../backend/director.py#L109) |
| S3 | "administer rocuronium" / "Nurse, proceed" **loops** the confirm question | No turn-spanning `pending_action` state; `parse_med` reads only the current utterance; no `confirm`/`proceed` op exists, so affirmations fall to `chitchat` | [nurse_brain.py:276](../backend/nurse_brain.py#L276), op list [nurse_brain.py:64](../backend/nurse_brain.py#L64) |

S3 reproduced identically for **high-alert meds** (rocuronium) **and disposition** (IVIG/steroids +
ICU admit) — so the fix must be generic, not medication-specific.

---

## 1 · Design principles (unchanged guardrails)

Carried from [live_sim_vision.md §2](../live_sim_vision.md):

1. **The engine is the single source of truth** — `SimulationEngine` owns every clinical fact, the
   clock, gating, transitions, and now **turn-spanning protocol state**. The LLM/voice layer never
   invents medicine *and never holds cross-turn state*.
2. **Constitution in code, flavor in data.** Per-actor behavioral invariants in code; persona/voice
   in case data.
3. **Director is the orchestrator** — a thin, ~80% deterministic stage manager. We keep our own
   Director rather than adopting ElevenAgents (see §2.1).
4. **Degrade, never break.** Every LLM/voice call fails soft.
5. **Deterministic where it counts.** Routing and protocol state are testable offline; the LLM seam
   is only for phrasing and genuine ambiguity.

### 1.1 · Why our own Director, not ElevenAgents

[ElevenLabs agent-transfer](https://elevenlabs.io/docs/eleven-agents/customization/tools/system-tools/agent-transfer)
is **sequential, single-active-agent** handoff (auth → banking → mortgage) — a customer-support funnel.
A resus bay is **concurrent presence** (nurse + patient + EMS + family all in the room at once), with
**engine-grounded truth** and **proactive/unprompted** speech (cues). ElevenAgents also wants to own
the LLM, turn-taking, and intent — competing with our engine. The
[AWS Nova Sonic multi-agent pattern](https://aws.amazon.com/blogs/machine-learning/scalable-voice-agent-design-with-amazon-nova-sonic-multi-agent-tools-and-session-segmentation/)
(server-side orchestrator → specialist sub-agents, real-time intent from speech, filler phrases,
parallel/async tools) maps far better — but with our twist: the **engine**, not the LLM, owns clinical
truth. **Use ElevenLabs as a TTS/STT supplier under our own Director.**

---

## 2 · Target architecture

```
          ┌──────────────────────────── browser (dumb I/O) ───────────────────────────┐
          │  mic → STT  →  POST /api/voice/turn {transcript, session, forced_addressee?}│
          │                         ↑ streaming TTS ← {speaker, reply}                  │
          └────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
   ┌─────────────────────────── DIRECTOR (server — the single router) ──────────────────┐
   │ 1. ADDRESSEE      who is this for?  forced chip → vocative → LLM classify → ask-back │
   │                   (no "else → patient"; unaddressed resolves by CONTEXT)            │
   │ 2. INTENT         what do they want?  engine-grounded ops + confirm/cancel_pending  │
   │ 3. PROTOCOL STATE run.pending_action — fill slots / confirm / cancel across turns   │
   │ 4. DISPATCH       SimulationEngine (clinical truth + now protocol state)            │
   │ 5. VOICE          the addressed CharacterAgent (constitution+flavor+EmotionalState) │
   │ 6. CUES           proactive unprompted beats                                        │
   │ 7. PERSIST        {speaker, role_label, addressee, text, ts}  ← real identity       │
   └─────────────────────────────────────────────────────────────────────────────────────┘
```

One endpoint, one pipeline, Director always in the loop.

---

## 3 · Phases

Each phase ships with its own tests (see §4) and leaves the system runnable.

### Phase 1 — Headless conversation test harness *(build first)*

Build the harness **before** the fixes so it (a) characterizes the current broken behavior with
red tests that prove S1–S3, and (b) becomes the regression net every later phase turns green.
Full design in §4. Deliverable: `tests/sim_scripts/` + `tests/test_live_sim_conversations.py`,
runnable with `pytest -k live_sim` — no browser, no audio, no real LLM calls.

### Phase 2 — Server-side routing unification (fixes S1, S2)

- **Collapse to one endpoint** `/api/voice/turn`. Fold `run_patient_turn`'s grounded-patient logic
  into `run_turn` so the patient is just another routed CharacterAgent.
- **Move `detectAddressee` to the server.** Browser sends raw transcript (+ optional forced chip).
  The Director's `resolve_addressee` does: forced chip → vocative match → LLM addressee classify →
  ask-back on low confidence. **Delete the `else → patient` default**; unaddressed resolves by
  context (last speaker, who holds the relevant beat).
- **Persist the real speaker.** Replace hardcoded `role:"nurse"` in `_resolve_turn` and `fire_cues`
  with the computed `speaker` + role label.
- Keep `/api/voice/llm-patient` as a thin deprecated alias during migration, then remove.

### Phase 3 — Turn-spanning action protocol (fixes S3: meds + admission)

- Add `run.pending_action = {op, target, params, status: awaiting_info | awaiting_confirm}` to the
  engine run; include it in `engine.snapshot()`/`restore()` (ties to F6 versioning).
- Add interpreter ops **`confirm_pending`** and **`cancel_pending`**; map bare affirmations
  ("yes", "proceed", "go ahead", "confirm") / negations to them so they stop falling to `chitchat`.
- `dispatch` **does not execute** a high-alert med or a confirmable action until status reaches
  `awaiting_confirm` → confirmed. Resolution rules:
  - follow-up supplying missing info (a dose, a ward) **fills the parked slot**, not a fresh parse;
  - bare affirmation **executes** the parked action and clears it;
  - a genuinely new order **supersedes** the parked one (nurse notes "holding the rocuronium, then").
- Generalize `needs_dose_confirm` into a `needs_confirmation(action)` predicate covering high-alert
  meds, disposition/admission, intubation prep, etc.

### Phase 4 — Transcript & session schema migration (S2 durability)

- Migration `008_voice_transcript.sql`: one ordered transcript representation —
  `{speaker, role_label, addressee, text, ts}` per turn — replacing the pre-multi-actor
  `history_json` + `patient_history_json` split.
- **Additive first** (add `turns_json` alongside the old two channels; backfill from them) to avoid
  stranding in-flight sessions, then cut over and drop the old columns in a later migration.
- Add `snapshot_version` to the engine snapshot (foundations F6) so parked runs survive engine
  shape changes — needed now that `pending_action` lives in the snapshot.

### Phase 5 — CharacterAgents first-class + EmotionalState ✅ DONE

- Constitutions consolidated into `backend/constitution.py` (keyed off `actors.SPEAKERS`).
- **EmotionalState computed live** — `backend/emotional_state.py` `for_speaker(kind, engine)`
  returns `(descriptor, instruction, stability)` from vitals/state: the patient escalates
  (ported 1:1 from the old `patient_voice` logic), a relative's worry tracks acuity, the
  professionals stay steady (`stability=None`). The instruction colours the prompt
  (`actor_voice._voice_as`, `patient_voice`); the stability rides the turn payload
  (`voice.py`) and the front-end `playTTS` for any speaker. The composition
  `Constitution + CaseFlavor + EmotionalState + VoiceBinding` is documented in
  `backend/agents.py`. Tests: `tests/test_emotional_state.py`.

### Phase 6 — ElevenLabs + latency realism ✅ DONE (one item deferred)

- **Flash v2.5** TTS (`eleven_flash_v2_5`), per-actor `voice_id` (env), streaming PCM-16000
  + barge-in, addressee chips — all in place. STT is ElevenLabs **Scribe** (`scribe_v1`).
- Latency UX: added a **dispatch filler** (`voice_room.html` `scheduleFiller`) — a brief
  spoken acknowledgement if a turn is slow (>850 ms), queued ahead of the real reply via the
  serialized speak queue; plus the TTS concurrency fix (one speak queue, no silent drops).
- **Deferred:** Scribe **v2 Realtime** (streaming websocket STT) — `scribe_v1` batch works and
  the realtime variant is a larger STT rewrite; revisit if STT latency becomes the bottleneck.
  Together soft-fallback was dropped earlier (ElevenLabs is the sole provider in this build).

### Sequencing

Phase 1 → 2 → 3 are the critical path (the harness, then the two structural fixes). 4 hardens
durability. 5 and 6 are the realism layer and depend on the unified pipeline existing. Do **not**
adopt ElevenAgents; ElevenLabs is a supplier only.

---

## 4 · The headless live-sim conversation test harness

**Goal:** assert how the *whole system* behaves across a multi-turn conversation — good runs and bad
runs — without the browser, audio, or live LLM calls. Run with `pytest -k live_sim` (CI-friendly,
seconds, deterministic).

### 4.1 · Seams (already deterministic)

- `dispatch(NurseIntent(...), engine, transcript)` — pure, no LLM (existing tests use this).
- `run_action(session, op, target)` — grounded turn, **skips the interpret LLM**.
- `run_turn(session, transcript)` — full pipeline; calls `interpret()` (LLM) + `speak()` (LLM).

To keep conversation tests deterministic we **stub the two LLM seams** and assert on structured
state, never on LLM prose:

- **Stub `interpret`** with a scripted `transcript → NurseIntent` mapping per conversation (so the
  script reads like real speech but routing/dispatch stay deterministic). Equivalent: drive
  `run_action`/`dispatch` directly with intents.
- **Stub `speak` / voice helpers** to echo a tagged marker (e.g. `"<nurse> …"`) so we can assert the
  **speaker** without depending on generated wording.

### 4.2 · Conversation script format

A script is data — a YAML/JSON list of turns plus expectations, stored in `tests/sim_scripts/`:

```yaml
case: myasthenic_crisis
name: gold_recognition_and_intubation
turns:
  - say: "EMS, what's the story?"
    expect: { speaker: ems, op: ask_history, reveals: ems_handover }
  - say: "Nurse, examine her neuro system"
    expect: { speaker: nurse, op: perform_exam, target: Neurological }
  - say: "Nurse, give rocuronium"
    expect: { speaker: nurse, pending: { drug: rocuronium, status: awaiting_info } }
  - say: "0.6 mig per kig"
    expect: { pending: { drug: rocuronium, dose: "0.6 mg/kg", status: awaiting_confirm } }
  - say: "Nurse, proceed"
    expect: { op: confirm_pending, administered: rocuronium, transitioned: post_intubation }
  - say: "Admit her to ICU on IVIG and steroids"
    expect: { speaker: nurse, transitioned: true }
final:
  reached_state: post_intubation
  working_dx_contains: "myasthenic"
  critical_error: false
  min_score: 70
```

The runner loads the script, builds a `VoiceSession` on the case (from the DB — see
[[gold-case-db-not-file]]), feeds each `say` through `run_turn` (with stubbed LLM), and asserts each
turn's `NurseTurn` (`speaker`, `intent.op`, `transitioned`, `critical_error`, `pending_action`) plus
the `final` block (terminal state, working dx, score from `sim_scoring`).

### 4.3 · The two script families the user asked for

**Gold conversations** (the resident does well): the case advances along the authored path, the right
actors speak, confirmations resolve cleanly, terminal state reached, score high. One per major
branch (recognition→intubation, the collateral-history path, the VC/NIF escalation path).

**Failing conversations** (the resident does *not* choose the correct path): assert the system
behaves *correctly in the face of poor performance* —
- **Wrong drug / standard succinylcholine** → engine trips the critical PEA-arrest (already modeled,
  [test_nurse_brain.py:113](../tests/test_nurse_brain.py#L113)); assert `critical_error`, the
  deterioration, and the low score.
- **Inaction** → drive the autonomous clock (`advance_clock`) with no intervention; assert the
  vitals modifiers fire (RR↑, SpO₂↓) and the case deteriorates on schedule.
- **Never confirms a high-alert dose** → assert the med is **not** administered and the loop does not
  occur (post-Phase-3 the pending action just sits `awaiting_confirm`, no re-loop).
- **Mis-addressing under ASR noise** → feed "Emma" where "EMS" was meant; assert the Director's
  ask-back fires instead of silently routing to the patient (the S1 regression guard).
- **Premature/incorrect diagnosis** → assert it's recorded neutrally, not graded mid-run, and the
  debrief reflects the miss.

### 4.4 · Characterization tests (Phase 1, run against current code)

Before any fix, encode S1–S3 as **expected-fail** assertions that prove the bugs, then flip them to
expected-pass as Phases 2–3 land. This is the regression net: the rocuronium loop and the
EMS→patient mis-route become permanent guards.

### 4.5 · What we deliberately do **not** test here

LLM wording, TTS audio, STT accuracy — those are supplier concerns with their own narrow tests
(`test_voice_service.py`). The conversation harness asserts **routing, state, transitions, and
scoring** only — the deterministic spine.

---

## 5 · Risks & guardrails

- **Don't let the LLM own state.** Routing and `pending_action` are engine/Director concerns; the
  LLM phrases and emotes only. This is the line that fixes S1–S3 for good.
- **Schema migration on live sessions.** Additive-first (Phase 4) so in-flight runs don't strand;
  snapshot versioning (F6) before `pending_action` ships in the snapshot.
- **Stubbed-LLM drift.** The harness stubs `interpret`; add a *small* set of un-stubbed smoke tests
  (a handful of utterances → real interpret) so the scripted mapping doesn't drift from reality.
- **Don't overfit to myasthenic crisis.** Scripts are per-case data; the harness and Director are
  case-agnostic. Add a second case's scripts before declaring generality.
```
