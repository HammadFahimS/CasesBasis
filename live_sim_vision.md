# CasesBasis Live Sim — Vision & Roadmap to a Finished Product

> A product + architecture roadmap for taking the voice-driven simulation from "works on the gold
> case" to "feels like a real, shipped clinical simulator." Written against what we have today
> (`SimulationEngine`, the `nurse_brain` turn pipeline, the `actors.py` cast registry) and what the
> **ElevenLabs grant** now unlocks (see [elevenlabs_capabilities.md](elevenlabs_capabilities.md)).
>
> **The governing tension:** make the **one gold case (myasthenic crisis) run flawlessly now**,
> **without overfitting** — every system must generalize to diverse future cases. The rule of thumb
> below is *"constitution in code, flavor in data, truth in the engine."*

---

## 1 · North star

A resident walks up to a deteriorating patient. They **talk** — to the patient, the nurse, EMS, the
husband, a consultant — and the **room talks back**, in distinct, emotionally-real voices, in real
time. The patient sounds genuinely breathless and gets worse if ignored. The nurse executes orders
and pushes back on unsafe ones. A consultant advises but won't fly the plane for you. The case moves
on a clock whether you act or not. At the end you get an honest, specific debrief. It should feel less
like a chatbot and more like **being in the resus bay**.

---

## 2 · Design principles (the guardrails that prevent overfitting)

1. **The engine is the single source of truth.** `SimulationEngine` owns every clinical fact, the
   clock, gating, transitions, vitals, and scoring. No LLM/voice layer may invent clinical content.
   This is what lets us add expressive AI *around* the sim without it hallucinating medicine.
2. **Constitution in code, flavor in data.** Each *kind* of actor (nurse, patient, EMS, family, RT,
   consultant) has invariant behavioral rules ("a **constitution**") that live in code and apply to
   **every** case. Each *case* layers a specific persona/voice/emotional arc on top, in the case data
   — hand-authored for gold, LLM-generated for the rest.
3. **Author-overridable, generator-fillable.** Anything the generator can write, an author can
   override. Gold case = fully hand-tuned and always wins; future cases = generated defaults that are
   "good enough," then optionally polished.
4. **Degrade, never break.** Every AI call (voice, persona, emotion) fails soft to a plain, correct
   fallback. The room must keep running if ElevenLabs/LLM hiccups.
5. **Latency is a feature.** A real resus bay has no 3-second pauses. Streaming STT/TTS and a tight
   turn loop are product requirements, not optimizations.

---

## 3 · The agent architecture (the core question: orchestrator + dynamic per-actor agents?)

**Short answer: yes to both — but lean.** We want a thin **Director** and a roster of **Character
profiles**, not a swarm of heavyweight autonomous agents each making their own decisions.

### 3.1 Three layers per character

```
CharacterAgent(for a case)  =  Constitution(kind)   ← code, invariant, every case
                             +  CaseFlavor(persona)  ← data, authored OR generated per case
                             +  EmotionalState(now)  ← derived live from engine state/vitals
                             +  VoiceBinding          ← ElevenLabs voice_id + voice_settings
```

- **Constitution (code, per speaker-kind).** The unchanging rules of the role. Today these are
  scattered (`VOICE_SYSTEM` in `actor_voice.py`, `PATIENT_SYSTEM` in `patient_voice.py`, the nurse
  rules in `nurse_brain`). **Consolidate them into one constitution registry**, keyed off
  `actors.SPEAKERS`. Examples of invariants:
  - *Nurse:* executes orders, stays bedside, confirms high-alert doses, never diagnoses for the
    resident, reports engine results verbatim.
  - *Patient:* answers only what's asked, never names their diagnosis, speaks in-character, can be
    too breathless to talk.
  - *Consultant:* advises and teaches, recommends, but **does not take over** management.
  - *Family:* gives collateral/emotional context, only what they'd plausibly know.
  - *EMS:* gives the prehospital handover, then recedes.
  - *RT:* performs/ reports bedside respiratory measures.
- **CaseFlavor (data, per case).** The specific human: name, backstory, speech style, emotional arc,
  what they know. This already has a home — `ActorRole.persona` / `.script` and
  `PatientProfile.persona` in `schemas/simulated_case.py`. **For the gold case, hand-write these.**
  **For future cases, the generator writes them** (see §5).
- **EmotionalState (derived live).** Not authored as a constant — *computed* from the engine each
  turn: a patient at SpO₂ 82 and a "Peri-arrest" state speaks slower, shorter, with `[gasping]`; a
  calm consultant stays measured. This is what makes voices **context-aware** rather than static.
- **VoiceBinding.** An ElevenLabs `voice_id` + per-actor `voice_settings`, driven by the emotional
  state (low `stability` for an anxious patient, etc.). See §4.

### 3.2 The Director (orchestrator) — thin, mostly deterministic

We **do** need an orchestration layer, but it should be a **stage manager**, not a brain that invents
plot. Its job each turn:

1. **Who speaks?** Resolve the addressee (already deterministic: `actors.kind_for_word`, unaddressed →
   patient). The Director owns this routing + the fallback ("husband not here yet → nurse covers").
2. **Should anyone speak unprompted?** Fire due cues / proactive actor beats (the cue engine already
   does this; the Director is its home). E.g. the monitor alarms, the nurse escalates, the patient
   deteriorates audibly.
3. **In what register?** Compute each speaking character's EmotionalState from engine state and hand
   it to their voice.
4. **Ground the content.** Pull the grounded line from the engine (history beat, result read, exam),
   then have the CharacterAgent *voice* it. **Never** let the character author clinical facts.

> **Crucially, the Director is NOT the engine and NOT an autonomous agent per actor.** It's a
> coordinator that's ~80% deterministic (routing, cues, gating) with a *small* LLM seam for genuine
> ambiguity (soft intent, emotional direction). This keeps behavior testable offline and prevents the
> "agents going rogue / inventing medicine" failure mode. It also generalizes: the same Director runs
> any case; only the cast and flavor change.

### 3.3 Do we spin up a live agent per actor?

**No long-running per-actor processes.** Model each actor as a **resolved profile** (constitution +
flavor + voice) that the Director invokes on demand to produce one grounded line. This is cheaper,
deterministic, and parks/resumes trivially with the session. (If we later adopt **ElevenAgents**, the
shape is "one voice-I/O agent + engine-as-tools + Director-driven voice switching" — see
[elevenlabs_capabilities.md §7](elevenlabs_capabilities.md). Treat that as a spike, not the baseline.)

---

## 4 · Emotion & context-aware voices (the realism unlock)

This is where ElevenLabs changes the felt experience. Map **engine state → voice**:

- **Per-actor voice identity.** Give each cast member a distinct ElevenLabs `voice_id` (via **Voice
  Design** from a text prompt, so it generalizes to any case). Store on `ActorRole.voice` (repurpose
  the existing field to hold an `voice_id`, or add `voice_id`). Replace the Cartesia name strings in
  `actors.SPEAKERS` / `TOGETHER_*_TTS_VOICE`.
- **Dynamic `voice_settings` from state.** Patient deteriorating → lower `stability` (more affect),
  slower `speed`, shorter lines. Consultant → high stability, measured. The Director computes these
  from `EmotionalState`.
- **Audio tags for set-pieces (Eleven v3).** For non-realtime or high-drama lines, inject tags:
  patient `[breathless]`/`[gasping]`, family `[worried]`, nurse `[urgent]` during a crash. (v3 isn't
  the low-latency model, so reserve tags for the opening handoff, the crash moment, the debrief — use
  Flash v2.5 + settings for the chatty turns.)
- **The deterioration arc you can hear.** As the myasthenic patient tires toward respiratory failure,
  the voice should *audibly* fade — fewer words, longer pauses (`...`), then unable to speak (the room
  switches to the nurse: "She's stopped talking, doctor"). This is a scripted **emotional state per
  scenario state**, generalizable to any deteriorating case.
- **Streaming TTS + barge-in.** Use the streaming/WebSocket TTS path so audio starts fast, and allow
  the resident to **interrupt** a long line (full-duplex feel). Huge realism win.

---

## 5 · Case authoring & generation (the anti-overfit engine)

To run gold flawlessly *and* scale to diverse cases, the **case generator** must produce everything
the live sim consumes — so a new case is "live-ready" without hand-tuning.

- **Generate the cast + flavor.** Extend the generator to author `ActorRole`s with persona/script and
  a **Voice Design prompt** per actor, so each new case ships distinct voices automatically.
- **Generate the emotional arc.** For each scenario state, author the patient's affect/voice register
  (calm → anxious → exhausted → silent), so deterioration is audible in any case.
- **Generate cues + scoring.** Proactive prompts and `MedicationExpectation`/criteria so the Director
  and debrief have material in every case (gold has these hand-authored).
- **A "case constitution" validator.** Before a case goes live, validate it has: ≥1 patient voice, a
  terminal state, at least one scored criterion, gated investigations with reads, and voice bindings.
  This is the safety net that keeps generated cases from feeling broken.
- **Gold-case overrides always win.** Hand-authored fields are never overwritten by generation (this
  already matches `seed_gold_cases` being insert-only — see the DB-vs-file gotcha in §8).

> **Net effect:** the gold case is hand-perfected; future cases inherit the same machinery with
> generated defaults. Nothing in the live runtime is hard-coded to myasthenic crisis.

---

## 6 · Live UX & stylistic polish (making it *feel* finished)

The room currently works; these make it feel like a product:

- **"Who's speaking" presence.** A live indicator/avatar that lights up for the current speaker
  (patient/nurse/EMS/…), color-coded to the Room log tags. Tie it to TTS playback start/stop.
- **Live captions.** Stream the spoken line as captions under the speaker (accessibility + clarity in
  a noisy room), with the addressee highlighted.
- **The monitor as a centerpiece.** We already have the Lead II monitor; extend to a proper bedside
  monitor: HR/SpO₂/BP/RR tiles that **react to engine vitals in real time**, alarm states (color +
  audible alarm) when out of range, flatline on PEA. Make it the visual anchor of the room.
- **Ambient soundscape.** Subtle ED ambience + monitor beeps tied to HR. Optional toggle. Cheap, huge
  immersion gain.
- **State/act transitions.** A tasteful title-card or status-chip animation when the case advances
  ("Recognition → Intubation"), so the resident feels the arc.
- **Addressee affordances.** Quick-target buttons/chips ("Patient · Nurse · EMS · Husband · Neuro")
  so the resident can direct a line without perfectly voicing the vocative; still allow free speech.
- **Polished entry.** The voice-backend pick modal, "tap to start," and the opening handoff should
  feel cinematic — a short briefing card, then the nurse's voice. First impressions sell the product.
- **Debrief as a report, not a dump.** Score donut, the assessment checklist (already built), a
  timeline of the run (what you did, when), result reads, and the spoken coaching note — laid out like
  a clinical OSCE feedback sheet. Shareable/exportable.
- **Consistent visual language.** Reuse the workbench design tokens (teal/ink palette, cards, chips)
  so the room matches the dashboard and feels like one app.

---

## 7 · Fixes & robustness (so gold runs flawlessly)

- **Reseed script.** Ship `scripts/reseed_case.py <case_id>` (or a `--reseed` flag) so gold-case JSON
  edits reliably reach the DB — closes the long-standing **DB-vs-file** footgun (the live sim reads
  `get_case` from the DB; `seed_gold_cases` is insert-only). *(See memory `gold-case-db-not-file`.)*
- **Voice fallback hardening.** Keep the existing soft-fallback (4xx → default voice) and extend it to
  provider-level fallback (ElevenLabs → Together) so the room never goes silent.
- **Latency budget + telemetry.** Measure end-to-end turn latency (STT→engine→TTS) and surface it in
  diagnostics; set a target (e.g. < 1.5s to first audio) and pick models to hit it.
- **Addressee mis-routes.** Add tests for the routing edge cases the walkthrough flags (vocatives mid-
  sentence, ambiguous "her"/"him", consult vs nurse). Make the Director's routing the one tested seam.
- **Cue timing.** Tune `CUE_AFTER_SECONDS` and ensure cues feel proactive but not naggy; once-per-cue
  is already enforced via `run.fired_cues`.
- **Interruption/turn-taking correctness.** When barge-in lands, cleanly cancel in-flight TTS and
  re-open the mic without dropping engine state.

### 7.1 Live playtest findings — 2026-06-25 (gold case, myasthenic_crisis)

A real run surfaced the following. Confirm each against current HEAD first (the running server may
predate recent routing work), then fix:

- [ ] **History spoken by the *nurse* in first person.** The authored `patient_symptoms` beat ("I sleep
  on three pillows…") was delivered with `speaker:"nurse"`, not the patient. By design
  (`nurse_brain._voice_reply` → `voice_history`) a beat whose provider is `patient` must be voiced by
  the **patient**; the DB case tags it `patient`, so the server returning `nurse` is the inverse of
  intended. **Most likely a stale server** (pre actor-voice routing) — restart on HEAD and re-test; if
  it persists, log per-turn `intent.op/target/speaker` to catch the misroute. Even when the nurse
  *relays*, she must use third person ("She reports…"), never "I".
- [ ] **Two unreconciled history paths.** History-taking is engine-grounded only on the **nurse channel**
  (`ask_history` reveals authored beats); the **patient channel** (`/api/voice/llm-patient` →
  `patient_reply`) is ungrounded free roleplay with only persona/known facts. So asking the *patient*
  directly yields thin distress lines while the authored timeline only surfaces via the nurse — the
  opposite of a resident's instinct. Ground the patient channel in the same beats (or route patient-held
  `ask_history` through the patient voice) so history-taking works whoever you talk to.
- [ ] **No "you already have this" handling.** Re-asking re-dumps the same beat (delivered twice,
  reworded) and the patient loops near-identical lines. The engine already tracks
  `run.revealed_history` — use it to acknowledge ("I already told you…") or add nothing on repeat.
- [ ] **Ambiguous/low-confidence orders resolve silently.** "perform an ACG" (typo/mishear) mapped to
  **ABG** with no confirmation; the resident likely meant ECG (asked for it next). Add a "did you
  mean…?" guard when the interpreter's match is low-confidence or ambiguous between named tests.
- [ ] **Case-data fix: ABG labeled but venous.** The ABG result reads "pO₂ 42 **venous**" — a venous gas
  isn't an ABG. Correct the authored investigation in the case (and reseed).
- [ ] **Flat clinical tension.** Crisis physiology (pH 7.31, pCO₂ 55, hypoxia) but no deterioration
  cue/state change and a static-repetitive patient affect across the run. Confirm `LIVE_RUNTIME` is on
  and wire the **emotional/deterioration arc** (Phase 1) so distress escalates with the clock.
- [ ] **Handoff reads like a chart, not speech.** Opening narration ("currently on 6 L via face mask
  with SpO₂ 97%, sitting upright and unable to speak full sentences") is HPI prose; tighten
  `INTRO_SYSTEM` toward spoken bedside handoff.
- [ ] **Actor self-voices collateral unprompted.** The husband volunteered family history ("my wife's
  mother had a neuromuscular disorder…") without being engaged — the proactive **cue system**
  (`_fire_cues` → `engine.due_cues`) firing an authored `family` cue on the clock. Decide the intended
  behavior: should an actor *self-voice* collateral before the resident addresses them, or should the
  cue only *nudge* ("the husband is here if you want to talk to him") and gate the actual beat behind
  the resident engaging? Tune cue authoring + `CUE_AFTER_SECONDS` accordingly.
- [ ] **ElevenLabs not wired (confirmed).** TTS still routes to Together AI Cartesia Sonic
  (`voice_service.synthesize` → `api.together.ai/v1/audio/speech`, `cartesia/sonic-2`);
  `ELEVENLABS_API_KEY` is in `.env` but unused — hence grant credits unchanged. This is the unstarted
  **Phase 1** swap, called out here so it isn't mistaken for a regression.

---

## 8 · Scoring & debrief depth

- **Richer criteria, still deterministic.** Keep `sim_scoring` pure; add criteria categories
  (recognition, safety/critical-error avoidance, correct interventions, dosing nuance, disposition,
  communication/CRM) so any case can be graded on the same rubric skeleton.
- **Per-case rubric authored in data.** The criteria definitions come from the case
  (`MedicationExpectation`, objectives, expected actions), so scoring generalizes.
- **Timeline + missed-opportunity feedback.** "You recognized the crisis at 4:10 but didn't check a
  VC until 7:30." Deterministic, drawn from the engine log.
- **Analytics that teach.** Trend the rubric categories across runs so a resident sees their *pattern*
  (e.g. consistently slow to escalate airway), not just a score.

---

## 9 · Phased roadmap

**Phase 1 — Gold polish + voice realism (now).**
- Consolidate the **constitution registry** (code) and wire it through `nurse_brain`/`*_voice`.
- ElevenLabs **TTS swap** (Flash v2.5) behind `voice_service.synthesize`; per-actor `voice_id`s via
  Voice Design; per-actor `voice_settings`.
- **Emotional state → voice** for the patient deterioration arc on the gold case.
- "Who's speaking" indicator + live captions + monitor alarms.
- Ship `reseed_case.py`. Harden voice fallback. Latency telemetry.

**Phase 2 — Director + emotional engine (generalize).**
- Extract the **Director** module (routing + cues + emotional-state computation) from `nurse_brain`.
  - **Uncertainty ask-back:** When the Director cannot confidently resolve the addressee from
    context, it asks the resident back: "Whom is this query for?" (short spoken prompt via the nurse
    voice, plus a brief UI disambiguation chip row). This prevents silent mis-routes during the
    early-Director window before chips exist.
- **Addressee affordances + multi-actor routing.** The Director owns "who speaks this turn?" with
  full context (last speaker, patient state, actor availability). Add quick-tap addressee chips
  ("Patient · Nurse · EMS · Husband · Neuro") to the UI so the resident can target an actor without
  voicing a vocative — the chip sets the forced addressee the Director reads. Shelf the routing-logic
  change until the Director exists; the chips are a Phase 2 frontend complement to it.
- Scribe **STT** (realtime + keyterm prompting on case drug/test names).
- Streaming TTS + **barge-in**.
- Ambient soundscape, state-transition polish, debrief-as-report.

**Phase 3 — Diverse cases at scale (anti-overfit payoff).**
- Generator authors cast + flavor + Voice Design prompts + emotional arcs + cues + criteria.
- Case-constitution **validator** gate before a case goes live.
- ElevenAgents **spike** (engine-as-tools) as an alternative voice-I/O runtime — evaluate, don't commit.

**Foundation hardening (carry through the phases, not a separate phase).** These are the
`foundations.md` F6–F8 items; they aren't features but they unblock the schema and agent growth above,
so schedule them deliberately as each phase touches the relevant surface.

- **F6 · Engine snapshot versioning 🟡** — `engine.snapshot()`/`restore()` is unversioned JSON, so a
  change to the engine's internal shape **breaks parked sessions on resume**. Add a `snapshot_version`
  + a tiny upgrade path *before* the Director/emotional-engine work (Phase 2) starts mutating engine
  state, so a deploy never strands in-flight runs.
- **F7 · Case-schema migration/backfill 🟡** — the cast/voice/emotion work adds fields to the case
  schema (`voice_id`, `voice_settings`, per-state emotional arcs on `ActorRole`), but each case is one
  stored `case_json` blob, so old rows lack the new fields. Add a **case-migration step** (the
  migrations-runner concept applied to `case_json` content) alongside `reseed_case.py` (F4), and keep
  Pydantic defaults generous so old blobs still validate. Land this **with** the first schema addition
  (Phase 1 `voice_id`/`voice_settings`) rather than after, so growth never silently breaks live cases.
- *(F8 · latency budget is already folded into the phases above — "Latency telemetry" in Phase 1 and
  the async/streamed turn chain in Phase 2.)*

> Throughline (from `foundations.md`): the **engine stays deterministic truth**; F6 keeps its parked
> state durable across deploys and F7 keeps its case data forward-compatible as the vision grows the
> schema. Neither is urgent on its own — both become urgent the moment the phase that needs them lands.

---

## 10 · Risks & guardrails

- **Don't let agents own medicine.** The Director/voices phrase and emote; the **engine grounds every
  fact**. This is the line that keeps a "more agentic" system safe.
- **Don't overfit to myasthenic crisis.** Anything case-specific lives in **data**, never in the
  runtime. The validator + generator are how we prove generality before we have many cases.
- **Latency vs expressiveness.** v3/audio-tags are gorgeous but slow — reserve for set-pieces; Flash +
  settings for live turns.
- **Cost.** Grant-sponsored, but keep chatty turns on Flash; reserve v3 for handoff/crash/debrief.
- **Complexity creep.** A thin Director beats a swarm of autonomous agents — fewer moving parts, fully
  testable, easier to reason about when a case misbehaves live.

---

## 11 · Open product decisions (for Hammad)

1. **Voice identities:** fixed per role (one nurse voice everywhere) or designed per case persona
   (richer, more to manage)? *Recommendation: fixed per role now, per-case for the patient.*
2. **Barge-in:** do we want true interruption in Phase 1, or is "start audio fast" enough at first?
3. **ElevenAgents:** worth a Phase-3 spike, or stay with our own Director + ElevenLabs TTS/STT?
4. **Ambient sound / captions:** in scope for the "finished product" feel, or post-launch polish?
5. **Scope of the gold-case emotional arc:** how scripted should the deterioration be (fully authored
   states vs computed from vitals)?
