from __future__ import annotations

import re

from agents.base import OpenAIJsonAgent
from agents.prompt_loader import load
from schemas.clinical_case import ClinicalCase


def build_patient_system_prompt(case: "ClinicalCase") -> str:
    persona = case.patient.persona_description.strip()
    occupation = case.patient.occupation
    age = case.patient.age
    gender = case.patient.gender

    occupation_str = f", {occupation}" if occupation else ""
    character_line = f"You are a {age}-year-old {gender}{occupation_str}."
    if persona:
        character_block = f"CHARACTER:\n{character_line}\n{persona}\nSpeak and behave exactly as this person would — your tone, vocabulary, and emotional register must match. The rules below still govern what information you share."
    else:
        character_block = f"CHARACTER:\n{character_line}\nSpeak as a real patient would."

    return f"You are a patient in a medical simulation for trainee doctors. Your role is to make the student work for every piece of information — not to help them.\n\n{character_block}\n\n{load('patient', 'rules')}"


class PatientResponseAgent(OpenAIJsonAgent):
    """Produces natural-language patient answers from case data."""

    def answer(
        self,
        case: ClinicalCase,
        question: str,
        conversation_history: list[tuple[str, str]] | None = None,
        revealed_symptoms: list[str] | None = None,
    ) -> str:
        sanitized_question = sanitize_question(question)
        if asks_for_diagnosis(sanitized_question, case.diagnosis):
            return "I'm not sure"
        prompt = build_patient_prompt(
            case, sanitized_question, conversation_history or [], revealed_symptoms or []
        )
        answer = self.text_response(
            system_prompt=build_patient_system_prompt(case),
            user_prompt=prompt,
        )
        return redact_diagnosis(answer, case.diagnosis)


def build_patient_prompt(
    case: ClinicalCase,
    question: str,
    history: list[tuple[str, str]],
    revealed_symptoms: list[str],
) -> str:
    parts = [f"Patient data:\n{case_to_patient_context(case)}"]

    if history:
        lines = "\n".join(
            f"Student: {q}\nYou: {a}" for q, a in history
        )
        parts.append(f"Conversation so far:\n{lines}")

    if revealed_symptoms:
        parts.append(
            f"Symptoms you have already mentioned: {', '.join(revealed_symptoms)}. "
            "Stay consistent with what you have said."
        )

    parts.append(f"Student: {question}\nAnswer as the patient.")
    return "\n\n".join(parts)


def case_to_patient_context(case: ClinicalCase) -> str:
    # Exclude diagnosis (hidden) and exam_findings (clinical signs the patient
    # cannot self-report — tachycardia, crepitations, etc. must be found by examination)
    return case.model_dump_json(exclude={"diagnosis", "exam_findings"}, indent=2)


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
