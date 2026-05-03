from __future__ import annotations

import re

from agents.base import OpenAIJsonAgent
from schemas.clinical_case import ClinicalCase

PATIENT_RESPONSE_SYSTEM_PROMPT = """
You are role-playing as the patient in a medical simulation.

Rules:
- Answer based only on the patient data provided.
- If the question is unrelated to the diagnosis, still answer if a normal patient could reasonably answer it.
- Do NOT expose or name the diagnosis directly.
- Use natural spoken language, not JSON.
- If the information is unknown, say "I'm not sure".
- Keep responses realistic and consistent with the case.
- Sound like a real patient, not a clinician.
- Occasionally be uncertain with phrases like "I think" or "maybe" when appropriate.
- Use approximate answers for non-critical details instead of exact clinical precision.
- Do not explain symptoms with perfect medical clarity.
- Vary tone and detail level across answers.
- For broad questions like "what symptoms do you have?", mention the main complaint and at most one or two associated details; do not list every symptom unless asked specifically.
- Do not mention that you are using a case file or simulation data.
""".strip()


class PatientResponseAgent(OpenAIJsonAgent):
    """Produces natural-language patient answers from case data."""

    def answer(self, case: ClinicalCase, question: str) -> str:
        sanitized_question = sanitize_question(question)
        if asks_for_diagnosis(sanitized_question, case.diagnosis):
            return "I'm not sure"
        prompt = f"""
Patient data:
{case_to_patient_context(case)}

User question:
{sanitized_question}

Answer as the patient.
""".strip()
        answer = self.text_response(
            system_prompt=PATIENT_RESPONSE_SYSTEM_PROMPT,
            user_prompt=prompt,
        )
        return redact_diagnosis(answer, case.diagnosis)


def case_to_patient_context(case: ClinicalCase) -> str:
    return case.model_dump_json(exclude={"diagnosis"}, indent=2)


def sanitize_question(question: str) -> str:
    sanitized = " ".join(question.split()).strip()
    if not sanitized:
        raise ValueError("question is required")
    return sanitized[:500]


def asks_for_diagnosis(question: str, diagnosis: str) -> bool:
    normalized_question = question.lower()
    normalized_diagnosis = diagnosis.strip().lower()
    if normalized_diagnosis and normalized_diagnosis in normalized_question:
        return True
    diagnosis_terms = (
        "diagnosis",
        "diagnosed",
        "what do you have",
        "what is wrong with you",
        "what's wrong with you",
        "what condition",
        "which condition",
        "what disease",
    )
    return any(term in normalized_question for term in diagnosis_terms)


def redact_diagnosis(answer: str, diagnosis: str) -> str:
    normalized_diagnosis = diagnosis.strip()
    if not normalized_diagnosis:
        return answer.strip()
    pattern = re.compile(re.escape(normalized_diagnosis), re.IGNORECASE)
    return pattern.sub("this condition", answer).strip()
