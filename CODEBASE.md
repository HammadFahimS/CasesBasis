# CODEBASE.md — CasesBasis

> Architectural map of the repo: what the major pieces are, how they fit together, and where
> to look first. For **setup & configuration** see [README.md](README.md); for full
> **implementation/handoff notes** see [claude_memory.md](claude_memory.md); for a **queryable
> knowledge graph** of the code see `graphify-out/` (`graphify query "<question>"`).

---

## 1. What this is

**CasesBasis** is an AI-powered clinical case simulator for trainee doctors ("residents").
It has grown two distinct product surfaces on top of one shared clinical core:

1. **Turn-based simulator** — generate a clinical case from a diagnosis, interrogate an AI
   patient, order investigations (ECG/CXR/labs), request a senior-doctor consult, then get a
   structured evaluation of your reasoning. (The original product; see `docs/ARCHITECTURE.md`.)
2. **Live voice ED simulation — "Room of Agents"** — a real-time, voice-driven emergency-department
   scenario with an autonomous clock. The resident speaks; a Router decides *who* in the room
   answers (patient, nurse, RT, EMS, family, phone consultants); time advances vitals and fires
   scripted cues; tasks given to actors report back after a latency. (The newest and largest body
   of work; see `docs/room-of-agents.md`, `docs/design_session_runtime.md`, `live_sim_vision.md`.)

Both surfaces share one rule: **the engine is the single source of clinical truth — no agent ever
invents a clinical value**, and the hidden diagnosis is never leaked to the learner.

---

## 2. Tech stack

| Concern | Choice |
|---|---|
| Language / runtime | Python 3.11 (conda env `casesbasis`; also pip-installable) |
| Web framework | FastAPI + Uvicorn, server-rendered HTML in `backend/static/` |
| Persistence | SQLite (per-thread connection, `BEGIN IMMEDIATE` writes) |
| Data contracts | Pydantic (strict models in `schemas/`) + a JSON Schema (`schemas/case_schema.json`) |
| LLM providers | Multi-provider behind one seam: Ollama (local), Groq, Together, OpenRouter, Cerebras, OpenAI |
| Voice | ElevenLabs TTS (the only TTS provider — a failure surfaces, never silently degrades); browser mic for STT |
| ECG | Deterministic SVG synthesizer **+** PTB-XL (PhysioNet) retrieval via a FAISS index |
| CXR | RoentGen-v2 dataset retrieval via FAISS |
| Knowledge graph | Per-session FalkorDB graph (optional; silently disabled if Docker is down) |
| Tracing | LangSmith (optional) |
| Tests | pytest (~49 test modules, offline by default via injected stubs) |

---

## 3. Directory map

```
agents/            LLM-backed agents (one provider-aware base, one class per role)
  prompts/         Markdown system/user prompts, loaded by prompt_loader.py
backend/           FastAPI app, all services, the engine, voice + live-sim layer
  routers/         FastAPI route modules (auth, cases, dashboard, pages, simulator, voice)
  static/          Server-rendered HTML pages + shared chrome/drawer partials
  utils/           Backend-local helpers
knowledge/         FalkorDB per-session clinical knowledge graph + local embedder
schemas/           Pydantic contracts + case_schema.json (the authored-case JSON schema)
cases/             Authored "gold" cases as JSON (e.g. myasthenic_crisis.json)
migrations/        SQL migrations applied by scripts/migrate.py
scripts/           Ops: seed, backup, migrate, reseed, build PTB-XL / CXR FAISS indexes
tests/             pytest suite
  sim_scripts/     YAML live-sim scenarios (failing_* and gold_* runs for the harness)
docs/              Architecture + the live-sim design docs (room-of-agents, session runtime, plan)
live-sim-ui-mehmood/  HTML UI mockups (SimRoom theme variants — exploratory, not wired)
docker-compose.yml FalkorDB container
environment.yml    Conda env spec   |   pyproject.toml  Package + clinical-case CLI entry point
```

---

## 4. The clinical spine

The graph's #1 god node (228 edges) is **`ClinicalCase`** (`schemas/clinical_case.py`) — the
source-of-truth schema every subsystem reads from. Around it:

- **Generation** — `ClinicalCaseSimulator` (`backend/service.py`) orchestrates: `GeneratorAgent`
  extracts concise medical facts → generates a `ClinicalCase` → `medical_validation.py` runs
  deterministic realism checks → on failure, `ValidatorAgent` escalates **correction → fallback →
  strict regeneration** (the "escalation ladder", see the validator prompts).
- **Authored cases** — richer, scripted scenarios live as JSON in `cases/` and are validated
  against `schemas/simulated_case.py` (`SimulatedCase`) / `case_schema.json`. They carry vitals,
  a progression timeline, actor roster (with `available_after_seconds`), cues, and outcomes.
- **Evaluation** — `ClinicalEvaluator` (`backend/clinical_evaluator.py`) scores symptom discovery,
  question relevance, diagnosis guess, and investigation interpretation into the `evaluation.py`
  schemas; `ReasoningReviewAgent` optionally refines the reasoning feedback;
  `performance_tracker.py` summarizes across sessions and recommends adaptive difficulty.

---

## 5. The agents (`agents/`)

All inherit the provider-aware base in `agents/base.py` (structured-JSON + text calls, safe
parsing, retries). Prompts live as markdown under `agents/prompts/` and are loaded by
`prompt_loader.py`.

| Agent | Role |
|---|---|
| `GeneratorAgent` | Extract medical facts, then generate a validated `ClinicalCase` |
| `ValidatorAgent` | Fallback correction / strict regeneration after a validation failure |
| `PatientResponseAgent` | In-character, **restrained** patient answers (only what's asked; no leaks) |
| `GenericInvestigationAgent` | Non-ECG/non-troponin investigation results (case-consistent) |
| `ConsultDoctorAgent` | Senior-doctor consult guidance from **redacted** context (no diagnosis) |
| `PhysicalExamAgent` | Exam findings, run through a diagnosis-leak guard (`diagnosis_guard.py`) |
| `ReasoningReviewAgent` | Optional refinement of `reasoning_feedback` only |
| `SimulationAuditorAgent` | Audits completed internal runs for QA findings |
| `StudentAgent` | Scripted/AI "student" that drives end-to-end self-play simulations |

LLM client plumbing: `backend/agents.py` (provider factory `build_llm_client` / `get_llm_provider`
/ `resolve_model`) with per-provider clients (`GroqClient`, `CerebrasClient` with model fallback,
Ollama/OpenAI/Together/OpenRouter builders) and `agent_runtime.py` (one instrumented `complete()`
seam wrapping every voice/character call, LangSmith-aware).

---

## 6. The live-sim layer ("Room of Agents")

This is the largest and most recent subsystem. Read `docs/room-of-agents.md` first — it states the
defect it fixes (the nurse was a universal fallback that laundered every routing failure into
plausible filler) and the **four contracts** that fix it:

- **C1 — Presence is first-class.** `SimulationEngine`/`RunState` own `present_actors`; addressing
  an absent actor gets a deterministic honest line, never silence or invented content.
- **C2 — Two-axis Router** (`backend/routing.py`): every utterance resolves to
  `Route(addressee, speech_act, target, …)`. *Who* and *what* are decided independently, killing
  the single-guess mis-route.
- **C3 — Bounded agents** (`backend/agents.py` `ROOM_AGENTS`): each actor answers only from the
  knowledge it holds and refuses in-character otherwise; **no agent answers for another.**
- **C4 — Hard diagnostics boundary**: visual studies (ECG/CXR) produce a *card* + a terse
  availability announcement; the interpretation is never read aloud (deterministic, not LLM hope).

Supporting modules:

- **`backend/session_runtime.py`** — `SessionRuntime`: the autonomous clock + `EventBus`. Ticks
  advance the scenario, persist state, and push `SessionEvent`s over a WebSocket to the client.
- **`backend/simulation_engine.py`** — `SimulationEngine` (god node #2): drives one live case via a
  `ScenarioState`/`RunState` state machine; owns presence, `due_cues`, `due_arrivals`,
  `due_deferred` (tasks that report back after a latency), and vitals transitions.
- **`backend/nurse_brain.py`** — classifies an utterance into a `NurseIntent` and `dispatch()`es
  deterministic engine ops (handles confirm/cancel, parked high-alert med doses).
- **`backend/director.py`** — owns cue firing, voice routing, ask-back phrasing, and the
  deterministic diagnostics announce (`_announce_study`, `fire_deferred`).
- **`backend/actors.py`** — the speaker-kind registry (`SPEAKERS`, `kind_for_word`, per-actor
  ElevenLabs voice + stability settings). `backend/emotional_state.py` tracks per-actor affect.
- **Voice seams** — `voice_session.py` (session lifecycle: `start_session`, access control),
  `actor_voice.py` / `patient_voice.py` / `consult_voice.py` (role-play phrasing per speaker),
  `voice_service.py` (ElevenLabs TTS: `synthesize`, streaming, sampling), `voice_debrief.py` +
  `sim_scoring.py` (post-run scoring against authored critical-error paths), `voice_persistence.py`.

YAML scenarios in `tests/sim_scripts/` (`failing_inaction`, `failing_premature_diagnosis`,
`failing_wrong_drug`, `gold_recognition_intubation`) drive the headless harness; live-sim guards
live in `tests/test_live_sim_conversations.py`.

---

## 7. Investigations, ECG & CXR

- **`backend/investigations.py`** — `InvestigationService` (god node #3) routes a requested test to
  deterministic builders (ECG, troponin) or the `GenericInvestigationAgent`, returning an
  `InvestigationResult`. Includes request sanitization and ischemic-feature detection.
- **ECG** — `ecg_synthesizer.py` produces a deterministic 12-lead inline SVG from `ECGParameters`;
  `ecg_service.py` serves it; `ecg_algorithm.py` emits a doctor-facing markdown trace of how the
  synthetic ECG was built. `ptbxl_service.py` (`PTBXLService` + `_PTBXLFaiss`) retrieves a real
  paper-style ECG image from PTB-XL by mapping the diagnosis to SCP codes and FAISS-searching.
- **CXR** — `cxr_service.py` (`CXRService`) retrieves chest films from the RoentGen-v2 dataset via a
  FAISS index. Index builders and the dataset downloader are in `scripts/`.
- **Embeddings** — `knowledge/embedder.py` (`LocalEmbedder`, HuggingFace, no API key required).

---

## 8. Web app, persistence & auth

- **`backend/web_app.py`** (+ `backend/main.py` entry, `web_deps.py` deps) — the FastAPI app;
  routers mounted from `backend/routers/` (`auth`, `cases`, `dashboard`, `pages`, `simulator`,
  `voice`). `pages.py` serves the HTML in `backend/static/` (home, analytics, simulator, voice room,
  CRUD case editor) with a shared chrome/drawer partial.
- **Auth** — `backend/auth.py`: resident accounts with **magic-link** login (`request_magic_link` /
  `consume_magic_link`) and a `require_resident` dependency guarding the voice/sim routes.
- **Persistence** — `backend/persistence.py` + `db.py` (per-thread SQLite, `write_transaction`);
  `dashboard.py` / `performance_tracker.py` read resident sessions, evaluations, and coaching
  insights. `simulated_case_repo.py` is the CRUD repository for authored cases, with **gold-case
  protection** (`CaseProtectedError` — gold cases can't be edited/deleted, only duplicated).
- **Case schema migrations** — `case_migrations.py` upgrades authored case JSON to the current
  schema version; `manage.py` is the management CLI (`python -m backend.manage`).

---

## 9. Knowledge graph (optional)

`knowledge/graph.py` (`ClinicalKnowledgeGraph`) maintains a **per-session FalkorDB** graph: it
extracts entities (symptoms, findings, meds, etc.) from each Q&A turn and investigation result in a
background thread, and feeds a formatted clinical summary back into the evaluator and doctor consult.
If FalkorDB/Docker isn't available it reconnects lazily and otherwise stays silently disabled.

---

## 10. Schemas (`schemas/`)

Strict Pydantic contracts (most extend a `StrictModel` that rejects unplanned fields):
`clinical_case.py` (`ClinicalCase`, `Patient`, `Outcome`, `ProgressionEvent`),
`simulated_case.py` (`SimulatedCase`, `InvestigationDef`, `VitalSigns`, actor roster, cues —
including the recently added presence/deferred-result fields), `evaluation.py`,
`investigation.py`, `doctor_consult.py`, `physical_exam.py`, `performance.py`,
`session_context.py`, `simulation.py`, `simulation_audit.py`, `account.py`, plus the authoritative
`case_schema.json`.

---

## 11. Key data flows

**Turn-based generation & evaluation**
```
diagnosis + difficulty
  → ClinicalCaseSimulator → GeneratorAgent (facts → case)
  → safe parse + normalize → medical_validation (deterministic)
  → [fail] ValidatorAgent (correct → fallback → strict regen)
  → ClinicalCase  →  patient Q&A / investigations / consult
  → ClinicalEvaluator → ClinicalEvaluation (+ ReasoningReviewAgent) → performance_tracker
```

**Live voice turn (Room of Agents)**
```
resident utterance
  → routing.route() → Route(addressee, speech_act, target)        # C2
  → present?  ─no→ honest-absent / summon line (router-voiced)    # C1
       │yes
  → ROOM_AGENTS[addressee].can_answer? ─no→ in-character refusal  # C3
       │yes
  → nurse_brain.dispatch → SimulationEngine op (clinical truth)
  → director: voice the reply / announce a study (card only)      # C4
  → SessionRuntime tick: advance clock, fire cues/arrivals/deferred, push over WS
```

---

## 12. Running it

```bash
# Web app (both surfaces)
uvicorn backend.main:app --reload --port 8000     # → http://localhost:8000

# One-shot turn-based generation (CLI entry point from pyproject.toml)
clinical-case "myocardial infarction" --difficulty medium
python -m backend.manual_demo

# Tests (offline by default)
pytest tests/ -v

# Ops
python -m backend.manage ...        # case migrations / reseed
python scripts/migrate.py           # apply SQL migrations
python scripts/seed.py              # dev residents + evaluations
python scripts/build_ptbxl_index.py # build the PTB-XL FAISS index
```

LLM provider is chosen with `LLM_PROVIDER` in `.env` (see README §Configuration). FalkorDB is
optional. ElevenLabs is required for live TTS but the turn-based surface works without it.

---

## 13. Where the bodies are (god nodes & bridges)

From the graphify analysis (`graphify-out/GRAPH_REPORT.md`):

- **`ClinicalCase`** (228 edges, betweenness 0.163) — the spine; touches generation, validation,
  evaluation, persistence, and every LLM client. Change it carefully.
- **`SimulationEngine`** (102 edges) — the live-sim hub: nurse brain, vitals transitions, scoring,
  routing, emotional state, narration all hang off it.
- **`InvestigationService`** (96) / **`InvestigationResult`** (91) — the test-ordering path.
- **`ClinicalEvaluation`** (94) / **`ClinicalEvaluator`** (88) — the scoring path.
- **`SimulatedCase`** — bridges the case store to CRUD protection, the runtime, and scenario state.

> This file is a human-oriented overview. For precise "what calls/uses X" answers, query the
> knowledge graph: `graphify query "how does the Router decide who answers"`.
