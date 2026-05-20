from __future__ import annotations

from agents.patient import (
    PatientResponseAgent,
    asks_for_diagnosis,
    case_to_patient_context,
    redact_diagnosis,
    sanitize_question,
)
from agents.prompt_loader import load
from backend.patient_response import PatientResponseService, normalize_case
from schemas.clinical_case import ClinicalCase
from tests.test_clinical_case_schema import sample_case


class StubPatientAgent(PatientResponseAgent):
    def __init__(self) -> None:
        self.last_system_prompt: str | None = None
        self.last_user_prompt: str | None = None

    def text_response(self, *, system_prompt: str, user_prompt: str) -> str:
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return "I feel feverish and dizzy, and I have not been peeing much."


def build_case() -> ClinicalCase:
    return ClinicalCase.model_validate(sample_case())


def test_patient_context_excludes_diagnosis() -> None:
    context = case_to_patient_context(build_case())

    assert "septic shock" not in context
    assert "fever" in context
    assert "blood_pressure" in context


def test_patient_agent_answers_from_case_context() -> None:
    agent = StubPatientAgent()

    answer = agent.answer(build_case(), "How are you feeling?")

    assert answer == "I feel feverish and dizzy, and I have not been peeing much."
    assert agent.last_user_prompt is not None
    assert "septic shock" not in agent.last_user_prompt
    assert "How are you feeling?" in agent.last_user_prompt


def test_patient_prompt_requests_realistic_nonclinical_variation() -> None:
    prompt = load("patient", "rules")
    assert "not a clinician" in prompt
    assert "imprecise" in prompt
    assert "Vary tone and detail level" in prompt
    assert "do not list every symptom" in prompt
    assert "Use lay language" in prompt


def test_patient_agent_does_not_answer_direct_diagnosis_question() -> None:
    agent = StubPatientAgent()

    answer = agent.answer(build_case(), "Were you diagnosed with septic shock?")

    assert answer == "I'm not sure"
    assert agent.last_user_prompt is None


def test_patient_service_accepts_raw_case_json() -> None:
    agent = StubPatientAgent()
    service = PatientResponseService(agent=agent)
    raw_case_json = build_case().model_dump_json()

    answer = service.answer_question(raw_case_json, "Do you have pain?")

    assert answer
    assert agent.last_user_prompt is not None


def test_normalize_case_accepts_dict() -> None:
    clinical_case = normalize_case(sample_case())

    assert clinical_case.diagnosis == "septic shock"


def test_sanitize_question_collapses_whitespace_and_limits_length() -> None:
    question = sanitize_question("  how\nare\tyou?  " + ("x" * 600))

    assert "\n" not in question
    assert "\t" not in question
    assert len(question) <= 500


def test_redact_diagnosis_removes_direct_diagnosis_mentions() -> None:
    answer = redact_diagnosis("They said it might be Septic Shock.", "septic shock")

    assert "Septic Shock" not in answer
    assert "this condition" in answer


def test_asks_for_diagnosis_detects_direct_questions() -> None:
    assert asks_for_diagnosis("What is your diagnosis?", "septic shock")
    assert asks_for_diagnosis("Do you have septic shock?", "septic shock")
