from __future__ import annotations

import re

from agents.base import OpenAIJsonAgent
from pydantic import BaseModel, ConfigDict, field_validator
from schemas.clinical_case import ClinicalCase
from schemas.evaluation import ClinicalEvaluation
from schemas.session_context import SessionContext, build_session_context

FACT_EXTRACTION_SYSTEM_PROMPT = """
You are extracting concise medical grounding facts for case generation.
Return valid JSON only.
Keep facts brief and medically standard.
Do not include treatment plans or explanations.
""".strip()

GENERATOR_PROMPT_TEMPLATE = """
You are a clinical case generator for a medical simulation system.

Your task is to generate a realistic, medically plausible patient case based ONLY on a diagnosis.

STRICT RULES:
- Output MUST be valid JSON only
- Do NOT include any explanations or extra text
- Do NOT break the schema
- Ensure medical consistency (symptoms, vitals, progression must align with diagnosis)
- Avoid contradictions
- Keep values realistic (no impossible vitals)

JSON STRUCTURE:

{
  "diagnosis": "",
  "patient": {
    "age": number,
    "gender": "male or female",
    "race": "one of: White, Black, Asian, Hispanic, Other",
    "occupation": "string or null",
    "persona_description": "string",
    "history": ["string"]
  },
  "symptoms": ["string"],
  "vitals": {
    "heart_rate": number,
    "blood_pressure": "systolic/diastolic",
    "respiratory_rate": number,
    "oxygen_saturation": number
  },
  "progression": [
    {
      "time_min": number,
      "description": "string"
    }
  ],
  "risk_factors": ["string"],
  "exam_findings": ["string"],
  "outcome": {
    "if_treated": "string",
    "if_untreated": "string"
  }
}

PERSONA GUIDELINES:
- Choose an age that is clinically realistic for the diagnosis and varies across cases (do not always default to middle-aged adults — use teenagers, young adults, elderly patients too)
- Set race to one of: White, Black, Asian, Hispanic, Other — vary it realistically across cases
- Set occupation consistent with age: omit (null) for patients under 14; use student, retired, or specific careers otherwise
- Write persona_description as exactly ONE sentence describing communication style and emotional state only
  Examples: "A shy 16-year-old who answers reluctantly and avoids detail."
            "A stoic retired farmer who minimises pain and resists fuss."
            "An anxious professional who over-explains and seeks reassurance."
            "A confused elderly man who struggles to recall exact timelines."
- persona_description must NOT describe information-sharing behaviour (never say 'freely shares', 'withholds', 'volunteers symptoms', etc.)
- Vary the persona meaningfully — not every patient is cooperative and articulate

MEDICAL GUIDELINES:
- Symptoms must strongly match the diagnosis
- Include at least 3-6 relevant symptoms
- Vitals must reflect severity (e.g., shock -> low BP, high HR)
- Progression must show worsening or change over time
- Include at least one realistic risk factor
- Avoid rare or irrelevant symptoms unless justified

PRESENTATION VARIATION:
- Do not always use the most classic textbook presentation
- Sometimes include atypical symptoms when they are medically plausible
- Sometimes make the case mild, early-stage, or still evolving
- Sometimes start with a misleading or non-specific complaint
- Keep the diagnosis medically supported by the overall case
- Keep difficulty balanced: avoid making the case either obvious from one clue or obscure without enough evidence

DIFFICULTY:
{difficulty_guidelines}

MEDICAL FACT GROUNDING:
Use these concise extracted facts as guidance. Do not copy them mechanically; keep the case realistic.
{medical_facts}

SESSION CONTEXT:
Use this only as a light bias. Do not overfit the case to it.
{session_context}

INPUT:
Diagnosis: {diagnosis}

Return JSON only.
""".strip()

MAX_DIAGNOSIS_CHARS = 160
MAX_GENERATION_ATTEMPTS = 2
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
DIFFICULTY_GUIDELINES = {
    "easy": """
- Use classic symptoms that strongly point to the diagnosis
- Make progression clear and easy to follow
- Include minimal misleading information
- Keep the case medically correct and straightforward
""".strip(),
    "medium": """
- Include some atypical or vague symptoms
- Allow moderate ambiguity while preserving enough diagnostic support
- Use a realistic mix of classic and non-specific findings
- Keep the case medically correct and balanced
""".strip(),
    "hard": """
- Use a subtle or misleading presentation
- Include plausible atypical symptoms
- Make progression delayed, early, or unclear
- Preserve enough evidence for the diagnosis without making it obvious
- Keep the case medically correct, not obscure for its own sake
""".strip(),
}


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
    facts_context = format_medical_facts(medical_facts)
    session_context_text = format_session_context(session_context)
    return GENERATOR_PROMPT_TEMPLATE.replace("{diagnosis}", sanitized_diagnosis).replace(
        "{difficulty_guidelines}",
        DIFFICULTY_GUIDELINES[normalized_difficulty],
    ).replace(
        "{medical_facts}",
        facts_context,
    ).replace(
        "{session_context}",
        session_context_text,
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
        prompt = f"""
Diagnosis: {sanitized_diagnosis}

Extract concise medical grounding facts:
- core symptoms
- possible atypical symptoms
- key risk factors

Return JSON only.
""".strip()
        return self.parse_response(
            system_prompt=FACT_EXTRACTION_SYSTEM_PROMPT,
            user_prompt=prompt,
            response_model=MedicalFacts,
        )
