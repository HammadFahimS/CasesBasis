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

1. **Physical Examination UI**: Add an "Examine Patient" button that reveals `exam_findings` from the case. Currently these are generated but inaccessible in the web UI.

2. **Session persistence**: Replace in-memory session dicts with a lightweight SQLite store so sessions survive server restarts.

3. **Better mobile voice**: Replace `webkitSpeechRecognition` with a Groq Whisper endpoint for cross-browser mic support.

4. **Richer ECG**: Tailor ECG waveforms to more diagnoses (PE → right heart strain, AF → irregular rhythm with no P waves, etc.).

5. **Scoring history**: Show students a history of their past case scores and areas to improve (the `PerformanceTracker` schema exists for this).

6. **Structured physical exam**: Beyond just revealing findings, allow the student to specify what they're examining (heart, lungs, abdomen) and return relevant findings selectively.
