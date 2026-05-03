# Repository Guidelines

This repository contains a Python/OpenAI clinical case simulator.

## Working Notes

- Prefer small, focused changes that match the project structure as it emerges.
- Keep successful simulator output as JSON only.
- The Pydantic schema in `schemas/clinical_case.py` is the source of truth for the case contract.
- The generator creates case JSON in one primary LLM call.
- Before case generation, the generator performs a transient medical fact grounding call for core symptoms, atypical symptoms, and risk factors. Do not persist these facts.
- Generation should include balanced presentation variation: atypical, mild, early, or misleading complaints when medically valid.
- Case generation supports `difficulty`: `easy`, `medium`, or `hard`; default is `medium`.
- `backend/medical_validation.py` performs deterministic post-generation medical checks using general body-system, severity, vital, and progression rules rather than disease-specific templates.
- The validator agent corrects generated JSON only after deterministic validation fails, limits edits to failed fields, and remains a fallback after repeated structured generation failure.
- `backend/patient_response.py` answers user questions from generated case JSON without exposing the diagnosis directly. Patient answers should sound realistic, uncertain when appropriate, and not overly clinical.
- `backend/investigations.py` routes investigation requests. ECG/EKG stays deterministic with structured findings and a schematic SVG; troponin/cardiac enzymes stay deterministic with numeric values and reference ranges; reasonable generic tests use `agents/investigation_generator.py`; nonsensical or impossible tests must return a safe unavailable response.
- `backend/clinical_evaluator.py` deterministically evaluates student questions, patient responses, and optional investigation interpretations against actual case data while avoiding disclosure of the diagnosis unless guessed. Include concise reasoning feedback, investigation interpretation feedback, and one targeted learning recommendation based on current misses plus optional prior results.
- Optional reasoning review may use `agents/reasoning_reviewer.py`, but only to replace `reasoning_feedback`; scores and all other evaluation fields must remain unchanged.
- `backend/performance_tracker.py` summarizes previous evaluation results passed as input and recommends adaptive difficulty without persistence.
- `schemas/session_context.py` summarizes `session_history` inputs only in memory. Use it as a light bias for generation and recommendations; do not store it or make it a hard template.
- Do not loosen schema strictness unless the JSON contract intentionally changes.

## Verification

- Install dev dependencies with `pip install -e .[dev]`.
- Run tests with `pytest`.
- For API-backed manual verification, set `OPENAI_API_KEY` and run `clinical-case "<diagnosis>"`.
