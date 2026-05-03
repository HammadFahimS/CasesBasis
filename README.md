# Clinical Case Simulator

Generate realistic educational medical case JSON from a diagnosis using a single primary LLM call.

`GeneratorAgent` first extracts concise medical facts for the diagnosis, then creates the case from the sanitized diagnosis plus those transient facts. It is prompted to vary presentations with plausible atypical, mild, early, or non-specific complaints while keeping the diagnosis medically supported. It retries generation up to two times if structured JSON parsing fails. The backend then runs deterministic medical realism checks using general body-system, severity, vital, and progression rules. If those checks fail, `ValidatorAgent` receives the generated JSON and returns a corrected case with the same structure.

`PatientResponseService` can answer user questions as the simulated patient from generated case JSON. It removes the direct diagnosis from the prompt context, answers in natural language, allows uncertainty and approximate non-critical details, and returns "I'm not sure" for direct diagnosis questions or unknown information.

`InvestigationService` returns investigation results from generated case JSON. ECG/EKG requests use a deterministic structured result with a schematic 12-lead SVG payload. Troponin/cardiac enzyme requests return deterministic numeric values with a reference range. Other reasonable clinical tests use a generic structured LLM path that keeps results student-interpretable without directly revealing the hidden diagnosis. Nonsensical or impossible requests return "That test is not available yet".

`DoctorConsultService` lets the student call a busy ER physician for quick guidance. It sends only redacted visible context to the LLM-backed consult agent, excluding the hidden diagnosis and outcome, and returns structured advice, missing information, safety warnings, and escalation status.

`ClinicalEvaluator` scores a student interaction from generated case JSON, student questions, patient responses, and optional investigation interpretations. It produces structured JSON covering diagnosis guess status, identified and missed symptoms, question relevance, investigation interpretation feedback, reasoning feedback, targeted learning recommendations, an overall 0-100 score, and brief feedback.

An optional LLM reasoning review can refine only `reasoning_feedback` for borderline or low-confidence evaluations. Scores and other structured fields remain deterministic.

`StudentPerformanceTracker` summarizes previous evaluation results passed in by the caller. It tracks recent scores, common missed symptoms, reasoning weaknesses, and recommends adaptive difficulty without requiring a database.

`session_history` can be passed as prior evaluation results. It is summarized in-memory to lightly bias future case generation toward repeated weaknesses, reinforce learning recommendation focus areas, and avoid repeating identical scenario patterns.

Successful simulator output is JSON only.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
Copy-Item .env.example .env
```

Ollama is the default LLM provider. Install/start Ollama locally and make sure the model in `.env.example` is available.

Default local configuration:

```powershell
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:7b
```

Ollama requests use its OpenAI-compatible endpoint at `http://localhost:11434/v1`.

To use OpenAI instead, explicitly set:

```powershell
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-2024-08-06
```

## Usage

```powershell
clinical-case "community-acquired pneumonia"
```

Write JSON to a file:

```powershell
clinical-case "acute pancreatitis" --output case.json
```

Choose difficulty:

```powershell
clinical-case "pneumonia" --difficulty hard
```

Run the full manual simulator flow:

```powershell
python -m backend.manual_demo
```

Run the simple web UI:

```powershell
uvicorn backend.web_app:app --reload
```

Then open `http://127.0.0.1:8000`.

Override the model:

```powershell
clinical-case "sepsis from pyelonephritis" --model qwen2.5:7b
```

## Project Structure

- `agents/`: provider-backed generator plus fallback recovery agent.
- `backend/`: orchestration service, patient response service, investigation service, doctor consult service, clinical evaluator, performance tracker, CLI, and non-LLM medical validation.
- `schemas/session_context.py`: input-driven session summary used to adapt generation and recommendations without persistence.
- `schemas/`: strict Pydantic JSON schema.
- `tests/`: focused schema and orchestration tests.

## Verification

```powershell
pytest
```

This tool is for educational simulation and does not provide medical advice.
