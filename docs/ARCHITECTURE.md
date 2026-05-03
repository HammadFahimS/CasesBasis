# Clinical Case Simulator Architecture

## Main Modules

- `agents/`: LLM-backed agents for case generation, fallback validation/correction, patient responses, generic investigation results, and optional reasoning review.
- `backend/service.py`: Orchestrates case generation, deterministic validation, and fallback correction.
- `backend/patient_response.py`: Wraps patient response generation and case normalization.
- `backend/investigations.py`: Routes requested tests to deterministic ECG/troponin results or generic LLM-backed investigation generation.
- `backend/clinical_evaluator.py`: Scores student questions, patient responses, diagnosis guesses, and investigation interpretations.
- `backend/medical_validation.py`: Deterministic medical realism checks after generation.
- `backend/performance_tracker.py`: Stateless performance summary and adaptive difficulty recommendation.
- `schemas/`: Strict Pydantic contracts for cases, evaluations, investigations, performance, and session context.
- `utils/common.py`: Shared small helpers for history normalization and most-common calculations.

## Data Flow

1. A caller requests a diagnosis, difficulty, and optional `session_history`.
2. `ClinicalCaseSimulator` calls `GeneratorAgent`.
3. `GeneratorAgent` first extracts concise medical facts, then generates a `ClinicalCase`.
4. LLM output is parsed through shared safe parsing and case normalization.
5. `backend/medical_validation.py` runs deterministic medical realism checks.
6. If validation fails, `ValidatorAgent` performs a second-pass correction.
7. The generated case can be used for patient questions, investigation ordering, and evaluation.
8. `ClinicalEvaluator` evaluates the interaction and produces structured feedback plus learning recommendations.

## LLM-Backed Components

- `GeneratorAgent`: medical fact extraction and primary case generation.
- `ValidatorAgent`: fallback recovery and correction after failed generation or validation.
- `PatientResponseAgent`: natural-language patient answers.
- `GenericInvestigationAgent`: non-ECG, non-troponin investigation results.
- `ReasoningReviewAgent`: optional refinement of `reasoning_feedback` only.

All LLM-backed components use the shared provider-aware base in `agents/base.py`.

## Deterministic Components

- Pydantic schemas in `schemas/`.
- Case output normalization and safe parsing.
- Medical realism validation in `backend/medical_validation.py`.
- ECG/EKG result generation and SVG payload.
- Troponin/cardiac enzyme result generation.
- Clinical evaluator scoring and structured feedback.
- Performance tracking and adaptive difficulty recommendation.
- Session context summarization.

## Validation

Validation happens in several layers:

- Schema validation: Pydantic models reject invalid structures and unplanned fields.
- Normalization before validation: common local-LLM formatting issues are cleaned, such as blood pressure and numeric vital strings.
- Generation retries: invalid JSON/schema failures retry within the generator.
- Deterministic medical validation: checks symptom consistency, vitals, contradictions, and progression.
- Correction fallback: `ValidatorAgent` only runs after validation failures or repeated structured generation failure.

## Offline Tests

Offline end-to-end tests avoid live LLM calls by injecting deterministic stubs:

- Fact extraction and case generation are mocked through a generator subclass.
- Patient responses are provided by a stub patient agent.
- ECG and troponin use deterministic investigation paths.
- Generic investigation generation is stubbed and fails if unexpectedly called.
- Optional reasoning review can be mocked with a reviewer stub.

This verifies the full simulator flow without requiring `OPENAI_API_KEY`, Ollama, or network access.

## Running With OpenAI

OpenAI is the default provider.

```powershell
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-2024-08-06
```

Run:

```powershell
clinical-case "myocardial infarction" --difficulty medium
python -m backend.manual_demo
```

## Running With Ollama

Ollama uses one shared local model for all LLM-backed components.

```powershell
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1
```

The simulator sends requests to Ollama's OpenAI-compatible endpoint:

```text
http://localhost:11434/v1
```

Run:

```powershell
clinical-case "myocardial infarction" --difficulty medium
python -m backend.manual_demo
```

## Process Summary

The simulator was built incrementally around a strict case schema. Each feature was added with a narrow implementation, deterministic validation where possible, and tests before broadening behavior. LLM use is isolated behind agents, while scoring, validation, ECG/troponin results, performance tracking, and session adaptation remain deterministic and testable.
