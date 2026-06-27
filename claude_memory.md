# CasesBasis — Claude Memory File

> Read this file first. It documents everything built in this project, how the system works, how to run it, and where to pick up. Written for a Claude assistant joining mid-stream.

---

## What This Project Is

**CasesBasis** is a clinical case simulator for trainee doctors. A student interacts with an AI patient, orders investigations, writes clinical interpretations, and attempts a diagnosis. The system evaluates the student's reasoning, tracks a knowledge graph of what they've discovered, and provides a senior clinician consult on demand.

It is a FastAPI single-page web application. All HTML, CSS, and JavaScript is served from one Python file (`backend/web_app.py`). No frontend build step, no separate React app.

---

## Repository Layout

```
CasesBasis/
├── agents/
│   ├── base.py                  # LLM client abstraction (Ollama, Groq, Together, OpenRouter, Cerebras, OpenAI)
│   ├── consult_doctor.py        # Senior doctor consult agent
│   ├── ecg.py                   # ECG waveform LLM agent (unused by web app — superseded by ECGService)
│   ├── generator.py             # Clinical case generator agent
│   ├── investigation_generator.py  # Generic investigation result generator (LLM-powered)
│   ├── patient.py               # Patient response agent (persona-aware)
│   ├── reasoning_reviewer.py    # Reasoning quality reviewer
│   ├── simulation_auditor.py    # Full simulation audit agent
│   ├── student.py               # Simulated student agent (for automated runs)
│   └── validator.py             # Case validation agent
│
├── backend/
│   ├── audit_fix_loop.py        # Automated audit-fix loop for simulation quality
│   ├── clinical_evaluator.py    # Evaluates student performance per session
│   ├── doctor_consult.py        # Doctor consult service (wraps consult_doctor agent)
│   ├── ecg_service.py           # ECG routing service (picks waveform by diagnosis)
│   ├── ecg_synthesizer.py       # SVG ECG waveform synthesizer (no external deps)
│   ├── investigations.py        # Investigation routing (ECG → synthesizer; else → LLM)
│   ├── main.py                  # Uvicorn entrypoint
│   ├── patient_response.py      # Patient response service (wraps patient agent)
│   ├── performance_tracker.py   # Tracks scores across sessions
│   ├── service.py               # High-level simulator service (case gen + eval)
│   ├── simulation_runner.py     # Automated simulation runner
│   └── web_app.py               # THE ENTIRE WEB APPLICATION (FastAPI + inline HTML/JS/CSS)
│
├── knowledge/
│   ├── embedder.py              # Embedding utilities (not used in web flow)
│   └── graph.py                 # ClinicalKnowledgeGraph — FalkorDB-backed per-session graph
│
├── schemas/
│   ├── clinical_case.py         # Pydantic models: ClinicalCase, Patient, Vitals, etc.
│   ├── doctor_consult.py        # DoctorConsultResult schema
│   ├── ecg.py                   # ECG visual payload schema
│   ├── evaluation.py            # ClinicalEvaluation schema
│   ├── investigation.py         # InvestigationResult schema
│   ├── performance.py           # Performance tracking schemas
│   ├── session_context.py       # SessionContext (cross-session learning bias)
│   └── simulation.py            # Simulation state schema
│
├── tests/                       # pytest test suite
├── utils/
│   ├── common.py                # Shared utilities
│   └── diagnosis.py             # Diagnosis name normalization
│
├── .env.example                 # All supported env vars with comments
├── claude_memory.md             # THIS FILE
├── docker-compose.yml           # FalkorDB container
├── environment.yml              # Conda environment (Python 3.11)
└── Makefile                     # Setup / start / stop shortcuts
```

---

## How to Run

### Prerequisites
- Python 3.11 (via conda) or any Python 3.11+ venv
- Docker (for FalkorDB knowledge graph — optional, degrades gracefully if absent)
- At least one LLM provider configured (Ollama is free and local)

### Quick Start

```bash
# 1. Clone the repo and enter it
cd CasesBasis

# 2. Create the conda environment (once)
make setup
# OR manually:
conda env create -f environment.yml
docker compose up -d   # start FalkorDB

# 3. Activate the environment
conda activate casesbasis

# 4. Copy and fill in your .env
cp .env.example .env
# Edit .env — see "Running with Ollama" below for a zero-API-key setup

# 5. Start the app
uvicorn backend.main:app --reload --port 8000
# OR
python -m backend.main

# 6. Open http://localhost:8000
```

---

## Running Fully Locally — Ollama (No API Keys)

This is the simplest setup. No cloud API keys needed.

### 1. Install Ollama

```bash
# macOS
brew install ollama
# or: https://ollama.com/download

# Start the Ollama server
ollama serve
```

### 2. Pull a model

```bash
# Default model the app uses:
ollama pull qwen2.5:7b

# Alternatives (better quality, more RAM):
ollama pull qwen2.5:14b
ollama pull llama3.1:8b
ollama pull mistral:7b
```

### 3. Configure `.env`

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=qwen2.5:7b

# Leave everything else blank — FalkorDB is optional too
FALKORDB_HOST=localhost
FALKORDB_PORT=6379
```

### 4. Run

```bash
conda activate casesbasis
uvicorn backend.main:app --reload --port 8000
```

The app starts. FalkorDB connection errors are logged as warnings but the app still works — the knowledge graph is just disabled.

---

## Supported LLM Providers

Set `LLM_PROVIDER` in `.env` to one of:

| Provider | Env vars needed | Notes |
|---|---|---|
| `ollama` | `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | Free, local, no API key |
| `groq` | `GROQ_API_KEY`, `GROQ_MODEL` | Fast free tier; `llama-3.3-70b-versatile` recommended |
| `together` | `TOGETHER_API_KEY`, `TOGETHER_MODEL` | Good for larger models |
| `openrouter` | `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` | Many models, some free |
| `cerebras` | `CEREBRAS_API_KEY`, `CEREBRAS_MODEL` | Very fast; Qwen-3 available |
| `openai` | `OPENAI_API_KEY`, `OPENAI_MODEL` | GPT-4o etc. |

The default model per provider is set in `agents/base.py`. Override via env vars.

---

## Application Features (Current State)

### 1. Case Generation
- Select a diagnosis from a dropdown (or type one) and difficulty (easy / medium / hard)
- LLM generates a full `ClinicalCase` JSON: patient, symptoms, vitals, progression timeline, risk factors, exam findings, outcome
- Case includes a **patient persona**: age, occupation, and a one-sentence communication style description
- Cases are cached for 5 minutes per (diagnosis, difficulty) pair to avoid duplicate API calls on refresh

### 2. Patient Interaction (Chat)
- Student types questions (or uses mic — see Voice I/O below)
- Patient agent responds in character, shaped by the case persona
- Patient is deliberately restrained: only answers what was directly asked, never volunteers extra symptoms
- Exam findings (tachycardia, crepitations etc.) are hidden from the patient — must be found via clinical exam (not yet implemented as a separate UI action, but the patient context excludes them)
- Patient does not reveal the diagnosis; if asked directly, says "I'm not sure"

### 3. Voice I/O
- **Mic input**: Uses browser `webkitSpeechRecognition` (Chrome desktop + Android). On mobile, triggers explicit `getUserMedia` permission request first.
- **TTS output**: Uses browser `speechSynthesis`. Gender-matched voice selection. Toggle button in chat header.
- Replay button on each patient message (hover to reveal) re-reads the text.
- If `webkitSpeechRecognition` is unavailable (Firefox, iOS), the mic button hides itself.

### 4. Investigations
- Student types an investigation request (e.g. "FBC, troponin, CXR, ABG")
- Multiple investigations parsed from comma/semicolon/ampersand/`and` separated input
- **ECG**: Routed to deterministic SVG waveform synthesizer — no LLM call. Displays as full-width panel below the main grid. Ischemic cases get ST elevation waveform; others get normal sinus rhythm.
- **All other investigations** (bloods, imaging, ABG, micro, echo, etc.): Routed to `GenericInvestigationAgent` which calls the LLM with the full case JSON. Returns structured results with numeric values, units, reference ranges, flags, and clinical hints.
- Blood tests include specific numeric values (not round numbers), flagged high/low/normal
- Cardiac biomarkers include serial testing notes when elevated
- ABG returns pH, PaO2, PaCO2, HCO3, base excess, SpO2 individually
- Investigation results never directly name the diagnosis

### 5. Clinical Knowledge Graph
- Per-session FalkorDB graph, backed by Docker container
- Every patient question+answer turn is processed in a background thread: LLM extracts clinical entities (Symptom, Medication, RiskFactor, Finding, Temporal, etc.) with description and negated flag
- Every investigation result is also processed
- Graph is seeded on case generation with the case's known symptoms and risk factors
- `graph.get_context(session_id)` → formatted text summary used as context in doctor consult
- `graph.get_entity_names(session_id)` → flat list of non-negated entity names used in evaluation
- Falls back gracefully (no-op, warning log) if FalkorDB is unavailable

### 6. Clinical Interpretations
- Student writes free-text interpretation of ordered tests
- Stored per session, fed into evaluation

### 7. Diagnosis Submission
- Student enters a differential diagnosis
- Used in final evaluation

### 8. Senior Doctor Consult
- Student can request a consult at any time
- Doctor consult agent receives: case (minus hidden diagnosis), conversation history, investigation results, graph context
- Returns a structured response: summary, key concerns, recommended next steps, teaching point

### 9. Evaluation
- Triggered when student completes the case
- `ClinicalEvaluator` scores symptom discovery (against graph entity names), reasoning quality, differential accuracy
- Returns scored feedback with detailed breakdown

### 10. Patient Overview Strip
Shows immediately after case generation:
- Chief complaint, Age, Gender, HR, BP, SpO2
- Persona row (indigo cards): Occupation + persona description (hidden if empty)

---

## Key Architecture Decisions

### Single-file web app
All HTML/CSS/JS lives in `backend/web_app.py` as a Python string. This was an intentional choice for simplicity — no build pipeline, no npm. The tradeoff is the file is large (~1500 lines). D3.js and Tailwind CSS are loaded from CDN.

### LLM client abstraction (`agents/base.py`)
`OpenAIJsonAgent` is the base class for all agents. It detects the provider from `LLM_PROVIDER` env var and routes to the right client. Ollama uses a raw HTTP client (no SDK dependency). Groq/Together/OpenRouter/Cerebras use the OpenAI SDK with a custom base URL. Structured JSON responses use schema-guided output (json_schema mode for Ollama; json_object + schema hint for Groq-style).

### Pydantic `extra="forbid"` + `normalize_case_data`
All schemas inherit `StrictModel` which rejects unknown fields. Because LLMs sometimes return messy JSON (floats instead of ints, extra whitespace), a `normalize_case_data` pre-processor runs before Pydantic validation to coerce types. This is the main resilience layer for generation failures.

### Knowledge graph is optional and async
`ClinicalKnowledgeGraph` writes happen in daemon background threads. The web handlers never wait for graph operations. If FalkorDB is down, everything degrades gracefully.

### ECG synthesizer
`backend/ecg_synthesizer.py` generates SVG ECG waveforms programmatically — no image files, no external service. It picks a waveform shape based on whether the case has ischemic features (STEMI → ST elevation; other → normal sinus). The SVG uses `viewBox` only (no fixed width/height) so it scales to any container.

---

## Files Changed in the Most Recent Development Session

These are the files that were actively modified (beyond their original state):

| File | What changed |
|---|---|
| `schemas/clinical_case.py` | Added `occupation: str \| None` and `persona_description: str` to `Patient` |
| `agents/generator.py` | Added PERSONA GUIDELINES section to prompt; added `occupation` and `persona_description` to JSON template |
| `agents/patient.py` | Replaced static system prompt with `build_patient_system_prompt(case)` — injects persona CHARACTER block dynamically |
| `agents/investigation_generator.py` | Heavily upgraded system prompt for blood tests (numeric values, flags), ABG, imaging, microbiology |
| `agents/base.py` | Added Cerebras and OpenRouter provider support |
| `backend/investigations.py` | Removed hardcoded troponin handler; all non-ECG investigations now go to `GenericInvestigationAgent` |
| `backend/web_app.py` | ECG full-width panel; voice I/O (mic + TTS); graph fit button; persona UI in vitals strip; graph wiring to evaluator + consult |
| `backend/clinical_evaluator.py` | Added `graph_entity_names` parameter to use discovered entities in symptom scoring |
| `backend/ecg_service.py` | New file — ECG routing service |
| `backend/ecg_synthesizer.py` | New file — SVG ECG waveform generator |
| `knowledge/graph.py` | New file — FalkorDB-backed per-session clinical knowledge graph |
| `schemas/ecg.py` | New file — ECG visual payload schema |
| `agents/ecg.py` | New file — ECG LLM agent (not used in main web flow) |
| `docker-compose.yml` | New file — FalkorDB container |
| `environment.yml` | New file — Conda environment spec |
| `Makefile` | New file — setup / start / stop shortcuts |

---

## Known Limitations / Things Not Yet Built

- **Examination findings**: The patient model has `exam_findings` (tachycardia, crepitations, etc.) but there is no UI button for "examine patient." These findings are hidden from the patient agent but accessible in the case JSON. A future "Examine" button should reveal them.
- **Voice on iOS**: `webkitSpeechRecognition` is not supported on iOS Safari or iOS Chrome. The mic button hides itself when unavailable. A Whisper-based backend route would solve this but was deferred.
- **Session persistence**: Sessions are in-memory Python dicts. Restarting the server loses all sessions. No database persistence for sessions yet.
- **Multi-user**: No authentication. The app is single-tenant by design — meant for one user at a time or a small group sharing a Cloudflare tunnel URL.
- **Self-hosting**: Cloudflare Tunnel was set up in this session to expose the local server. The user runs `cloudflared tunnel --url http://localhost:8000` to get a public URL. No dedicated infra.
- **ECG for non-cardiac cases**: ECG returns a normal sinus rhythm SVG for non-ischemic cases. It's functional but not highly tailored to e.g. a PE case showing right heart strain.
- **Graph UI**: D3.js force graph visualizes entities discovered per session. Nodes are clamped to viewport. A "Fit" button recenters. This works but can feel sluggish with many nodes.

---

## How the Patient Persona Works (Implementation Detail)

1. **Generation**: `agents/generator.py` prompts the LLM to fill `patient.occupation` and `patient.persona_description` in the case JSON. The persona guidelines instruct: vary ages broadly, omit occupation for under-14s, describe communication style and emotional tone in one sentence, never describe information-sharing willingness.

2. **Patient agent**: `agents/patient.py` — `build_patient_system_prompt(case)` creates a per-call system prompt with a `CHARACTER:` block:
   ```
   CHARACTER:
   You are a 16-year-old female.
   A shy teenager who answers reluctantly and avoids eye contact.
   Speak and behave exactly as this person would — your tone, vocabulary,
   and emotional register must match. The rules below still govern what
   information you share.
   ```
   Followed by the standard restraint rules (answer only what was asked, 1–3 sentences, no diagnosis, etc.)

3. **UI**: After case generation, the Patient Overview strip shows an indigo "Persona" row with occupation and the one-sentence description. Hidden if the case has no persona data (backwards compatibility).

---

## Data Flow (Web App)

```
POST /api/cases
  → GeneratorAgent.generate()         # LLM generates ClinicalCase JSON
  → ClinicalKnowledgeGraph.init_session()   # seeds graph with symptoms/risk factors (bg)
  → returns case_preview (age, gender, occupation, persona_description, vitals, chief_complaint)

POST /api/sessions/{id}/questions
  → PatientResponseAgent.answer()     # LLM responds in persona
  → ClinicalKnowledgeGraph.add_turn() # extracts entities from Q+A (bg)
  → returns {answer: str}

POST /api/sessions/{id}/investigations
  → InvestigationService.get_results()
    → ECG: ECGService.build_visual_payload() + build_ecg_result()
    → else: GenericInvestigationAgent.generate()   # LLM
  → ClinicalKnowledgeGraph.add_investigation()     # extracts from result (bg)
  → returns list[InvestigationResult]

POST /api/sessions/{id}/doctor-consult
  → DoctorConsultService.consult()
    → graph.get_context() → session_context for LLM
  → returns DoctorConsultResult

POST /api/sessions/{id}/evaluate
  → ClinicalEvaluator.evaluate()
    → graph.get_entity_names() → symptom scoring context
  → returns ClinicalEvaluation
```

---

## Environment Variables Reference

```env
# Provider selection
LLM_PROVIDER=ollama   # ollama | groq | together | openrouter | cerebras | openai

# Ollama (local, free)
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=qwen2.5:7b

# Groq
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.3-70b-versatile

# Together
TOGETHER_API_KEY=...
TOGETHER_MODEL=meta-llama/Llama-3.3-70B-Instruct-Turbo

# OpenRouter
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=nvidia/nemotron-3-super-120b-a12b:free

# Cerebras
CEREBRAS_API_KEY=...
CEREBRAS_MODEL=qwen-3-235b-a22b-instruct-2507

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-2024-08-06

# FalkorDB (optional — app works without it)
FALKORDB_HOST=localhost
FALKORDB_PORT=6379

# LangSmith tracing (optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=...
LANGCHAIN_PROJECT=casesbasis
```

---

## Running Tests

```bash
conda activate casesbasis
pytest tests/ -v

# Skip tests that hit real LLMs (fast, offline)
pytest tests/ -v -m "not llm"
```

---

## Recommended Next Steps

1. **Session persistence**: Sessions now persist via SQLite (`casesbasis.db`). Status: shipped.

2. **Better mobile voice**: Replace `webkitSpeechRecognition` with a Groq Whisper endpoint for cross-browser mic support.

3. **Richer ECG**: Tailor ECG waveforms to more diagnoses (PE → right heart strain, AF → irregular rhythm with no P waves, etc.). Status: substantially shipped — see Workbench v3 notes below.

4. **Scoring history**: Show students a history of their past case scores and areas to improve (the `PerformanceTracker` schema exists for this).

5. **Wire the Consults panel** (`panel-consults`): Neurology / Nephrology / ICU cards exist in the UI as TBA. Backend route + a `ConsultsService` to call a `consult_specialist` agent, then flip cards to `idle` state.

---

## Workbench v3 — Current UI Architecture (read this before touching `web_app.py`)

The web UI was rewritten into a GridStack workbench. The old horizontal patient strip and the old left/right docks are gone. Everything you see in the browser is in `backend/web_app.py` inside the `HTML_PAGE` string. All CSS lives in one `<style>` block; all JS in one `<script>` block.

### Layout flow

```
<body>             ← display:flex; flex-direction:column; min-height:100vh
  <header.cb-topbar>  ← sticky topbar with brand, diagnosis dropdown, generate, submit, etc.
  <main>           ← flex:1; min-height:0; display:flex; flex-direction:column
    <div.wb-workbench>  ← flex:1; left rail | #gs-area (GridStack grid) | right rail
  <footer.cb-statusbar>
```

`.wb-workbench` used to have `height: calc(100vh - 220px)` — a magic constant. **Do not bring that back.** It now uses flex so the layout fills any viewport.

### Rails (vertical tabs in the workbench)

**Left rail (YOU):** Interview · Physical Exam · Investigations.
**Right rail (DATA):** Imaging · Consults.

Performance and Knowledge Graph used to be rail tabs but were moved to the **status bar** as `.cb-sb-tab` chips, left of "CasesBasis · clinical simulation". They use the same `data-panel` / `wbTogglePanel` plumbing as rail tabs; only the styling differs.

The `Reasoning` panel exists but has **no rail tab**. Entry point: clicking **Submit** in the topbar. `showSubmitConfirm()` checks `#diagnosisGuess` — if empty, it opens `panel-reasoning`, toasts "Enter your working diagnosis to submit", and focuses the input.

### Panels (`#gs-store` → GridStack)

Configured in `WB_PANELS` (top of the workbench JS). Each `id` matches an off-grid `<section class="wb-float-panel" id="panel-…">` in `#gs-store`. `wbOpenPanel(id)` reparents the section into the grid; `wbClosePanel(id)` puts it back. `wbPinPanel(id)` adds to `_pinnedPanels` set; pinned panels are skipped by the "close-others" sweep `wbOpenPanel` runs.

| Panel | Notes |
|---|---|
| `panel-patient` | **Always pinned, auto-opens on case generate/resume.** Vertical column anchored top-right (`x:9, y:0, w:3, h:14`). Contains identity, persona, chips, chief complaint, 2×2 vitals, and the live Lead II ECG monitor. No rail tab — it is permanent UI. No close button. |
| `panel-interview` | Patient chat. |
| `panel-phys-exam` | 9 system cards with tinted backgrounds (redesigned per `design_physical_exam` handoff bundle). `EXAM_COLORS` map drives `sysColor` per system — **do not remove the `sysColor` declaration in `renderExamResult`**, the result-card template literal depends on it. |
| `panel-invest` | Free-text + quick-order chips (ABG, CBC, Troponin, D-dimer, U&E, Peak Flow). **CXR and ECG were removed from here** — they live in the Imaging panel. |
| `panel-reasoning` | Diagnosis input, interpretation, doctor consult. No rail tab — entered via Submit preflight. |
| `panel-imaging` | 5 modality cards (`.img-card`): CXR, 12-Lead ECG (active); Ultrasound, CT, MRI (TBA). State machine via `data-state` = `idle` / `pending` / `available` / `disabled`. **Green border = available.** `setImagingCardState(modality, state)` + `onImagingCardClick(modality)` handle state. CXR/ECG viewers (`panel-cxr`, `panel-ecg`) are no longer auto-opened on result — the user clicks the available card to open them. |
| `panel-consults` | TBA cards for Neurology / Nephrology / ICU. UI-only for now. |
| `panel-cxr` / `panel-ecg` | Viewer panels — kept, but no rail tabs. Opened from Imaging cards. |
| `panel-kg` | Knowledge graph viewer. Opened from status-bar chip. **No auto-open on case generate** (intentional). |
| `panel-perf` | Performance report. Opened from status-bar chip on demand, and auto-opens on case submit only. **No auto-open on case start** — the `wbOpenPanel('panel-perf')` call was deliberately removed; only `perfStatusBadge` is primed. |

### Patient panel live Lead II ECG monitor

- The dark `.pp-ecg` strip at the bottom of `panel-patient` is **hidden by default** (`id="ppEcgWrap" hidden`). It appears only after the resident orders an ECG investigation and a result arrives.
- Driven by `_initPatientEcg()` (canvas + RAF loop) and `setPatientPanelEcgMorphology(m)`. Uses Gaussian-sum morphology mirroring `backend/ecg_synthesizer._beat_template` (P/Q/R/S/T centers + LBBB/RBBB/delta variants + ST offset + T-wave inversion).
- The backend ships `lead_ii_morphology: LeadIIMorphology | None` on `ECGVisualPayload` (see `schemas/investigation.py`). `ECGService.build_visual_payload` populates it from `params.leads["II"]` plus PR/QRS/QT intervals and rhythm label. PTB-XL shelved branch leaves it `None`.
- Respects `prefers-reduced-motion` (renders a single static cycle).
- `_stopPatientEcg()` cancels the RAF on new case.
- `setPatientPanelHR(hr)` retunes the rhythm spacing without touching morphology.

### ECG synthesizer

- `_PTBXL_DISABLED = True` in `backend/ecg_service.py` — the LLM + numpy synthesizer is the primary path. PTB-XL code is intact for re-enable.
- 12-lead output now includes a **4th-row Lead II rhythm strip** spanning 10 seconds (`RHYTHM_DURATION = 10.0`, `RHYTHM_LEAD = "II"` in `ecg_synthesizer.py`). Separate `mingrid_rh`/`majgrid_rh` SVG patterns because the rhythm strip uses a different X scale than the 3×4 panels.

### Invariants to preserve

- `populateCasePreview(cp)` calls `wbOpenPanel('panel-patient')` + `wbPinPanel('panel-patient')` + `setPatientPanelHR` + `_stopPatientEcg()` + hides `#ppEcgWrap`. Don't drop those — the Patient panel won't appear, and the live monitor won't reset between cases.
- `_patientHR`, `_patientAge`, `_patientGender` are module-scoped JS vars. The ECG and CXR result handlers read them for overlay metadata (`RATE … bpm`, `M · 58`). They replaced the old DOM reads `pvHR`/`pvAge`/`pvGender` from the deleted strip.
- `.cb-ps-label` CSS rule is kept (used by perf panel "Overall Score" label) even though the rest of the old strip CSS was deleted.
- New-case reset path includes: `resetImagingPanel()`, `wbClosePanel('panel-imaging')`, `_stopPatientEcg()`, hide `#ppEcgWrap`. Keep these aligned if you add new per-case state.

### Files touched in the v3 work

- `backend/web_app.py` — bulk of changes.
- `backend/ecg_service.py` — `LeadIIMorphology` extraction + PTB-XL shelf flag.
- `backend/ecg_synthesizer.py` — 4th-row Lead II rhythm strip.
- `schemas/investigation.py` — added `LeadIIMorphology` and `lead_ii_morphology` field on `ECGVisualPayload` (optional, backwards compatible).
- `backend/diagnosis_guard.py` — central scrub for diagnosis-leakage prevention.
- `backend/static/_chrome.html`, `_drawer.html`, `home.html`, `analytics.html` — Home/analytics pages were extracted from the monolith.

### Design handoff bundles

Located outside the repo, in `/Users/hammad/Documents/CasesBasis/`:
- `design_handoff_dashboard_redesign/` — Home page redesign.
- `design_physical_exam/casesbasis/project/` — Physical Examination panel redesign + chats + screenshots.
- `design_patient_panel/casesbasis/project/` — Patient Panel design including `Patient Panel.html` (Lead II ECG canvas prototype). The ECG monitor code in `web_app.py` lifts this prototype's `ecg(phase)` Gaussian sum + sweep-monitor logic, then extends it with backend-driven morphology params.

When a new design bundle arrives, the README inside it explains: read the chat transcripts first, then the named HTML file.

---

## Live Voice Sim — the dynamic case engine (read this before touching `backend/voice_*` / `nurse_brain` / `actors`)

A second, separate experience from the Workbench v3 single-encounter UI: a **real-time,
voice-driven ED simulation**. The resident speaks; a cast of voices (patient, nurse, EMS,
family, RT, consultant) speaks back; a **state machine** drives the case forward on a
**wall-clock**. Entry point is `GET /simulator/voice-room` (`backend/static/voice_room.html`);
the case is **myasthenic crisis** (the gold case). See [docs/live-sim-walkthrough.md](docs/live-sim-walkthrough.md)
for a full step-by-step run that exercises every feature.

> **CRITICAL gotcha (see auto-memory `gold-case-db-not-file`):** the live sim loads the case
> from the **DB** via `get_case`, **not** `cases/*.json`. `seed_gold_cases()` is insert-only and
> never overwrites an existing row. Editing `cases/myasthenic_crisis.json` is **invisible** to
> the running app until the DB row is re-seeded, and you must start a **fresh** session
> (parked/resumed sessions keep their old case). Verify through `get_case`, not `load_seed_case`.

### Core engine — `backend/simulation_engine.py`
- `SimulationEngine` drives one encounter from a `schemas.simulated_case.SimulatedCase`. Owns
  mutable run state (current state, clock, revealed facts, critical errors, working diagnoses).
- **Hybrid trigger resolution:** deterministic triggers (`time` / `inaction` / `vitals` guard)
  fire on their own; `action` triggers match by **keyword** (offline default) **or** an injected
  `intent_matcher(action_text, transition_desc) -> bool` for soft LLM intent ("opts for intubation").
- Serialisable (`snapshot` / `restore`) → a run parks in the DB and resumes.
- **Cue engine:** `due_cues()` fires at most one authored cue per call once `elapsed_in_state ≥
  CUE_AFTER_SECONDS` (75s). Each cue fires once; `run.fired_cues` survives park/resume.

### Session lifecycle — `backend/voice_session.py`
- `VoiceSession` binds an engine + DB-loaded case + transcripts (separate `history` nurse channel
  and `patient_history` patient channel) + diagnostics `cards`. In-memory LRU (`_MAX_LIVE_SESSIONS=256`)
  backed by `voice_persistence` (every session is parked to the DB, rehydrates on demand — eviction
  is lossless).
- `advance_clock(session)` advances the engine by **real elapsed time** between turns, so the world
  moves while the resident deliberates. **Parked time does NOT advance the clock** (a parked patient
  must not deteriorate — `_rehydrate` resets `last_tick` to now; a concluded run is frozen).

### The turn pipeline — `backend/nurse_brain.py` (`run_turn`)
1. `advance_clock` — time/inaction transitions fire.
2. `interpret` — **one LLM call** → structured `NurseIntent` (engine op + target), chosen from the
   case's real catalogue of beats/investigations/actions.
3. `dispatch` — **pure, deterministic**, runs the op on the engine. No LLM → unit-tested offline.
4. `speak` — **one LLM call** phrases the nurse's reply from the engine's real outcome.
- Ops: `read_vitals`, `ask_history`, `order_investigation`, `perform_exam`, `order_medication`,
  `perform_action`, `request_consult`, `diagnose`, `chitchat`.
- All LLM access goes through a single `_complete` seam that tests monkeypatch.

### The cast / addressee routing — `backend/actors.py` (single source of truth)
- `SPEAKERS` registry of **speaker kinds** (`nurse`/`patient`/`ems`/`family`/`rt`/`consult`): each
  has synonyms (vocatives), a TTS voice env var + default, and `holds_history`. This unifies what
  used to be restated in four places (provider→speaker map, vocative table, TTS voice map).
- **Addressing:** lead an utterance with a name ("Nurse… / Husband… / EMS… / RT… / Neuro… / ICU…")
  to reach staff; **anything unaddressed goes to the patient.** `kind_for_word` / `speaker_voice`.
- It governs only *who voices* a line and *with what voice* — the engine remains the source of every
  clinical fact; nothing in `actors` invents content.

### Who speaks each line
- `backend/patient_voice.py` — free-form patient role-play (restrained: answers only what's asked).
- `backend/actor_voice.py` — **grounded** bystander voices: `voice_history` (EMS/husband/patient speak
  an authored history beat without inventing), `voice_bedside` (RT reports VC/NIF), `voice_cue` (turns
  an authored cue stage-direction into a natural spoken prompt). Distinct from `consult_voice` (a
  specialist who *advises*).
- TTS via `voice_service.synthesize` (Cartesia Sonic voice names; falls back to nurse voice if rejected).

### Gating & the medication mechanic (myasthenic crisis specifics)
- **Real-time gates:** results return only after wall-clock delay — **VC/NIF 60s · VBG 120s · husband
  arrives 300s · CT chest 300s.** Order it, keep working, ask again later.
- **Dose-confirm loop** (`backend/medication.py` — `needs_dose_confirm`, `parse_med`): ordering a
  high-alert drug with no dose makes the nurse ask you to confirm; correct paralytic (rocuronium,
  reduced dose) advances state; succinylcholine standard-dose → **critical error → PEA arrest** branch.

### Scoring & debrief
- `backend/sim_scoring.py` — `score_run` emits a flat list of `Criterion` (recognition, avoided
  critical errors, non-depolariser, reduced roc dose, avoided pyridostigmine). **Pure/deterministic,
  no LLM**, fails soft. New scoring lives here, not bolted onto features.
- `backend/voice_debrief.py` — `build_debrief`: settles the clock, then assembles (1) deterministic
  `run_summary` from the engine, (2) the case author's `teaching`, (3) the `score_run` assessment
  checklist, (4) one fail-soft LLM `coaching_note`. The deterministic debrief always renders.
- **Analytics loop:** voice runs surface in `/analytics` with a 🎙 Voice badge and score on the trend
  chart; `GET /api/dashboard/voice-review/{session_id}` backs the Review overlay.

### Voice-sim routes (`backend/web_app.py`)
`/api/voice/session` (start) · `/resume` · `/stt` · `/llm` (nurse turn) · `/action` · `/llm-patient`
(patient turn) · `/debrief` · `/pause` + `/unpause` · `/sessions` (list) + `DELETE /sessions/{id}` ·
`/tts` · `/sample`.

### Files in the live-sim work (current `feature/workbench-v3-ui` branch, uncommitted)
- `backend/actors.py` (new) + `tests/test_actors.py` (new) — the cast registry.
- `backend/actor_voice.py`, `backend/nurse_brain.py`, `backend/patient_voice.py`,
  `backend/voice_debrief.py`, `backend/web_app.py`, `backend/static/voice_room.html` — refactored to
  the single cast registry + cue/scoring/dosing work.
- `cases/myasthenic_crisis.json`, `schemas/simulated_case.py`, `casesbasis.db` — case content + schema
  (remember: DB is the live source, re-seed after JSON edits).
- Open TODO (from auto-memory): add `scripts/reseed_case.py <case_id>` (or a `--reseed` flag) so the
  DB re-seed isn't a hand-written SQL `UPDATE` each time.

---

## Quick orientation for the next Claude session

1. **All UI lives in `backend/web_app.py`.** Search there first.
2. **`pytest -q` runs 320 tests in ~30s.** Keep them green.
3. **Layout is flex top-to-bottom.** Do not reintroduce viewport-height arithmetic.
4. **The Patient panel is permanent.** Code paths that "reset on new case" must reset its child state, not close the panel.
5. **CXR/ECG viewers are no longer auto-opened on result.** The Imaging panel is the launch point. If you change this, update both the result handler and the user's expectations.
6. **Reasoning panel has no rail tab.** Submit is its entry point. Don't add a rail tab back without confirming.
7. **Performance and KG chips live in the status bar.** Their `data-panel` attribute means they reuse all rail-tab plumbing for free.

6. **Structured physical exam**: Beyond just revealing findings, allow the student to specify what they're examining (heart, lungs, abdomen) and return relevant findings selectively.
