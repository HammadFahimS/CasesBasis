from __future__ import annotations

import re

from agents.base import OpenAIJsonAgent
from agents.prompt_loader import load, render
from pydantic import BaseModel, ConfigDict, field_validator
from schemas.clinical_case import ClinicalCase
from schemas.evaluation import ClinicalEvaluation
from schemas.session_context import SessionContext, build_session_context

MAX_DIAGNOSIS_CHARS = 160
MAX_GENERATION_ATTEMPTS = 2
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


class MedicalFacts(BaseModel):
    """Concise diagnosis facts used to ground case generation."""

    model_config = ConfigDict(extra="forbid")

    core_symptoms: list[str]
    possible_atypical_symptoms: list[str]
    key_risk_factors: list[str]

    @field_validator("core_symptoms", "possible_atypical_symptoms", "key_risk_factors")
    @classmethod
    def keep_facts_concise(cls, value: list[str]) -> list[str]:
        return [item.strip()[:80] for item in value[:5] if item.strip()]


class GenerationError(RuntimeError):
    """Raised when case generation fails after configured retries."""

    pass


def sanitize_diagnosis(diagnosis: str) -> str:
    sanitized = re.sub(r"[\x00-\x1f\x7f]", " ", diagnosis)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    sanitized = sanitized[:MAX_DIAGNOSIS_CHARS].strip()
    if not sanitized:
        raise ValueError("diagnosis is required")
    return sanitized


def normalize_difficulty(difficulty: str) -> str:
    normalized = difficulty.strip().lower()
    if normalized not in VALID_DIFFICULTIES:
        raise ValueError("difficulty must be one of: easy, medium, hard")
    return normalized


def build_generation_prompt(
    diagnosis: str,
    difficulty: str = "medium",
    medical_facts: MedicalFacts | None = None,
    session_context: SessionContext | None = None,
) -> str:
    sanitized_diagnosis = sanitize_diagnosis(diagnosis)
    normalized_difficulty = normalize_difficulty(difficulty)
    return render(
        load("generator", "user"),
        difficulty_guidelines=load("generator", f"difficulty_{normalized_difficulty}"),
        medical_facts=format_medical_facts(medical_facts),
        session_context=format_session_context(session_context),
        diagnosis=sanitized_diagnosis,
    )


def format_medical_facts(medical_facts: MedicalFacts | None) -> str:
    if medical_facts is None:
        return "No extracted facts provided."
    return medical_facts.model_dump_json(indent=2)


def format_session_context(session_context: SessionContext | None) -> str:
    if session_context is None:
        return "No prior session context provided."
    if not any(
        [
            session_context.repeated_missed_symptoms,
            session_context.repeated_reasoning_weaknesses,
            session_context.learning_focus_areas,
            session_context.recent_scores,
        ]
    ):
        return "No meaningful prior session patterns detected."
    return f"""
- Slightly emphasize opportunities to practice: {session_context.learning_focus_areas or ["general clinical reasoning"]}
- Include chances to discover commonly missed symptom areas when medically appropriate: {session_context.repeated_missed_symptoms or ["none"]}
- Support practice on reasoning weaknesses: {session_context.repeated_reasoning_weaknesses or ["none"]}
- Avoid repeating an identical scenario; vary age, risk history, presenting complaint, symptom order, and progression style.
- Recent scores for context only: {session_context.recent_scores}
""".strip()


class GeneratorAgent(OpenAIJsonAgent):
    """Generates validated clinical case JSON from a diagnosis."""

    def generate(
        self,
        diagnosis: str,
        difficulty: str = "medium",
        session_history: list[ClinicalEvaluation | dict[str, object] | str] | None = None,
    ) -> ClinicalCase:
        sanitized_diagnosis = sanitize_diagnosis(diagnosis)
        medical_facts = self.extract_medical_facts(sanitized_diagnosis)
        session_context = build_session_context(session_history)
        prompt = build_generation_prompt(
            sanitized_diagnosis,
            difficulty,
            medical_facts,
            session_context,
        )
        errors: list[Exception] = []

        for _ in range(MAX_GENERATION_ATTEMPTS):
            try:
                generated_case = self.parse_response(
                    system_prompt="Return valid JSON only.",
                    user_prompt=prompt,
                    response_model=ClinicalCase,
                )
                if generated_case.diagnosis.strip().lower() != sanitized_diagnosis.lower():
                    generated_case.diagnosis = sanitized_diagnosis
                return generated_case
            except Exception as exc:
                errors.append(exc)

        raise GenerationError(
            f"case generation failed after {MAX_GENERATION_ATTEMPTS} attempts"
        ) from errors[-1]

    def extract_medical_facts(self, diagnosis: str) -> MedicalFacts:
        sanitized_diagnosis = sanitize_diagnosis(diagnosis)
        return self.parse_response(
            system_prompt=load("generator", "fact_system"),
            user_prompt=render(load("generator", "fact_user"), diagnosis=sanitized_diagnosis),
            response_model=MedicalFacts,
        )
