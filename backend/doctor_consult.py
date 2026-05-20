from __future__ import annotations

import re
from typing import Any

from agents.base import build_llm_client, get_llm_provider
from agents.consult_doctor import ConsultDoctorAgent, sanitize_consult_question
from backend.patient_response import normalize_case
from schemas.clinical_case import ClinicalCase
from schemas.doctor_consult import DoctorConsultResponse
from schemas.investigation import InvestigationResult


class DoctorConsultService:
    """Provides redacted, LLM-backed doctor consultation guidance."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str | None = None,
        agent: ConsultDoctorAgent | None = None,
    ) -> None:
        if agent:
            self.agent = agent
            return
        provider = get_llm_provider()
        shared_client = client or build_llm_client(provider)
        self.agent = ConsultDoctorAgent(
            client=shared_client,
            model=model,
            provider=provider,
        )

    def consult(
        self,
        case: ClinicalCase | dict[str, Any] | str,
        student_questions: list[str],
        patient_responses: list[str],
        investigations_ordered: list[InvestigationResult],
        consult_question: str,
        *,
        difficulty: str | None = None,
        session_context: dict[str, Any] | None = None,
    ) -> DoctorConsultResponse:
        clinical_case = normalize_case(case)
        sanitized_question = sanitize_consult_question(consult_question)
        redacted_context = build_redacted_consult_context(
            clinical_case,
            student_questions,
            patient_responses,
            investigations_ordered,
            difficulty=difficulty,
            session_context=session_context,
        )
        response = self.agent.consult(redacted_context, sanitized_question)
        return redact_hidden_diagnosis(response, clinical_case.diagnosis)


def build_redacted_consult_context(
    case: ClinicalCase,
    student_questions: list[str],
    patient_responses: list[str],
    investigations_ordered: list[InvestigationResult],
    *,
    difficulty: str | None = None,
    session_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    transcript = build_transcript_text(student_questions, patient_responses)
    context: dict[str, Any] = {
        "patient": case.patient.model_dump(),
        "presenting_complaint": case.symptoms[0] if case.symptoms else "",
        "discovered_symptoms": discover_items(case.symptoms, transcript),
        "vitals": case.vitals.model_dump(),
        "discovered_exam_findings": discover_items(case.exam_findings_flat, transcript),
        "investigations_available": [
            investigation_context(result) for result in investigations_ordered
        ],
    }
    if difficulty:
        context["difficulty"] = difficulty
    if session_context:
        context["session_context"] = session_context
    return redact_text_values(drop_hidden_context_keys(context), hidden_terms(case.diagnosis))


def build_transcript_text(student_questions: list[str], patient_responses: list[str]) -> str:
    return " ".join([*student_questions, *patient_responses]).lower()


def discover_items(items: list[str], transcript: str) -> list[str]:
    if not transcript.strip():
        return []
    discovered: list[str] = []
    for item in items:
        normalized = item.lower()
        words = [word for word in re.findall(r"[a-z]+", normalized) if len(word) >= 4]
        if normalized in transcript or any(word in transcript for word in words):
            discovered.append(item)
    return discovered


def investigation_context(result: InvestigationResult) -> dict[str, Any]:
    return {
        "investigation": result.investigation,
        "status": result.status,
        "structured_result": result.structured_result,
        "summary": result.summary,
    }


def redact_hidden_diagnosis(
    response: DoctorConsultResponse,
    diagnosis: str,
) -> DoctorConsultResponse:
    redacted = redact_text_values(response.model_dump(), hidden_terms(diagnosis))
    return DoctorConsultResponse.model_validate(redacted)


def drop_hidden_context_keys(value: Any) -> Any:
    hidden_keys = {"diagnosis", "outcome", "if_treated", "if_untreated"}
    if isinstance(value, list):
        return [drop_hidden_context_keys(item) for item in value]
    if isinstance(value, dict):
        return {
            key: drop_hidden_context_keys(item)
            for key, item in value.items()
            if str(key).lower() not in hidden_keys
        }
    return value


def redact_text_values(value: Any, terms: list[str]) -> Any:
    if isinstance(value, str):
        redacted = value
        for term in terms:
            if len(term.strip()) >= 2:
                redacted = re.sub(
                    hidden_term_pattern(term),
                    "the leading concern",
                    redacted,
                    flags=re.IGNORECASE,
                )
        return redacted
    if isinstance(value, list):
        return [redact_text_values(item, terms) for item in value]
    if isinstance(value, dict):
        return {
            key: redact_text_values(item, terms)
            for key, item in value.items()
        }
    return value


def hidden_terms(diagnosis: str) -> list[str]:
    normalized = diagnosis.strip()
    terms = [normalized] if normalized else []
    aliases = {
        "myocardial infarction": ["heart attack", "MI"],
        "community-acquired pneumonia": ["pneumonia"],
    }
    for key, values in aliases.items():
        if normalized.lower() == key:
            terms.extend(values)
    return terms


def hidden_term_pattern(term: str) -> str:
    escaped = re.escape(term)
    if re.fullmatch(r"[A-Za-z0-9 ]+", term):
        return rf"\b{escaped}\b"
    return escaped
