from __future__ import annotations

import pytest

from agents.generator import (
    GenerationError,
    GeneratorAgent,
    MedicalFacts,
    build_generation_prompt,
    normalize_difficulty,
    sanitize_diagnosis,
)
from agents.prompt_loader import load
from schemas.clinical_case import ClinicalCase
from schemas.session_context import SessionContext
from tests.test_clinical_case_schema import sample_case


class RetryingGenerator(GeneratorAgent):
    def __init__(self, failures_before_success: int) -> None:
        self.failures_before_success = failures_before_success
        self.calls = 0
        self.fact_calls = 0

    def parse_response(self, **_: object) -> ClinicalCase:
        self.calls += 1
        if self.calls <= self.failures_before_success:
            raise ValueError("invalid JSON")
        return ClinicalCase.model_validate(sample_case())

    def extract_medical_facts(self, diagnosis: str) -> MedicalFacts:
        self.fact_calls += 1
        return sample_facts()


class CapturingGenerator(GeneratorAgent):
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def parse_response(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        response_model = kwargs["response_model"]
        if response_model is MedicalFacts:
            return sample_facts()
        return ClinicalCase.model_validate(sample_case())


class BloodPressureRetryGenerator(GeneratorAgent):
    def __init__(self) -> None:
        self.calls = 0
        self.fact_calls = 0

    def parse_response(self, **_: object) -> ClinicalCase:
        self.calls += 1
        payload = sample_case()
        if self.calls == 1:
            payload["vitals"]["blood_pressure"] = "one twenty over eighty"
        else:
            payload["vitals"]["blood_pressure"] = "120 over 80"
        return ClinicalCase.model_validate(payload)

    def extract_medical_facts(self, diagnosis: str) -> MedicalFacts:
        self.fact_calls += 1
        return sample_facts()


def sample_facts() -> MedicalFacts:
    return MedicalFacts(
        core_symptoms=["fever", "confusion", "hypotension"],
        possible_atypical_symptoms=["weakness"],
        key_risk_factors=["diabetes"],
    )


def test_sanitize_diagnosis_removes_control_characters_and_limits_length() -> None:
    diagnosis = sanitize_diagnosis(" septic\nshock\t" + ("x" * 300))

    assert "\n" not in diagnosis
    assert "\t" not in diagnosis
    assert len(diagnosis) <= 160


def test_generator_retries_once_after_invalid_json() -> None:
    generator = RetryingGenerator(failures_before_success=1)

    clinical_case = generator.generate("septic shock")

    assert generator.calls == 2
    assert generator.fact_calls == 1
    assert clinical_case.diagnosis == "septic shock"


def test_generator_fails_after_two_invalid_json_attempts() -> None:
    generator = RetryingGenerator(failures_before_success=2)

    with pytest.raises(GenerationError):
        generator.generate("septic shock")

    assert generator.calls == 2
    assert generator.fact_calls == 1


def test_generator_prompt_requests_realistic_presentation_variation() -> None:
    prompt = load("generator", "user")
    assert "Do not always use the most classic textbook presentation" in prompt
    assert "atypical symptoms" in prompt
    assert "mild, early-stage, or still evolving" in prompt
    assert "misleading or non-specific complaint" in prompt
    assert "Keep difficulty balanced" in prompt


def test_build_generation_prompt_applies_easy_difficulty() -> None:
    session_context = SessionContext(
        repeated_missed_symptoms=["dizziness"],
        repeated_reasoning_weaknesses=["Clarify timing and progression of symptoms."],
        learning_focus_areas=["illness timeline clarification"],
        recent_scores=[62, 68],
    )
    prompt = build_generation_prompt(
        "pneumonia",
        difficulty="easy",
        medical_facts=sample_facts(),
        session_context=session_context,
    )

    assert "Diagnosis: pneumonia" in prompt
    assert "Use classic symptoms" in prompt
    assert "minimal misleading information" in prompt
    assert "core_symptoms" in prompt
    assert "fever" in prompt
    assert "SESSION CONTEXT" in prompt
    assert "illness timeline clarification" in prompt
    assert "Avoid repeating an identical scenario" in prompt


def test_build_generation_prompt_applies_medium_difficulty() -> None:
    prompt = build_generation_prompt("pneumonia", difficulty="medium")

    assert "some atypical or vague symptoms" in prompt
    assert "moderate ambiguity" in prompt


def test_build_generation_prompt_applies_hard_difficulty() -> None:
    prompt = build_generation_prompt("pneumonia", difficulty="hard")

    assert "subtle or misleading presentation" in prompt
    assert "delayed, early, or unclear" in prompt


def test_normalize_difficulty_rejects_invalid_value() -> None:
    with pytest.raises(ValueError):
        normalize_difficulty("expert")


def test_fact_extraction_prompt_is_concise_and_structured() -> None:
    prompt = load("generator", "fact_system")
    assert "Return valid JSON only" in prompt
    assert "Keep facts brief" in prompt
    assert "Do not include treatment plans" in prompt


def test_generator_extracts_facts_before_case_generation() -> None:
    generator = CapturingGenerator()

    clinical_case = generator.generate("septic shock")

    assert clinical_case.diagnosis == "septic shock"
    assert len(generator.calls) == 2
    assert generator.calls[0]["response_model"] is MedicalFacts
    assert generator.calls[1]["response_model"] is ClinicalCase
    assert "MEDICAL FACT GROUNDING" in str(generator.calls[1]["user_prompt"])
    assert "hypotension" in str(generator.calls[1]["user_prompt"])


def test_generator_retries_after_unrecoverable_blood_pressure_validation_failure() -> None:
    generator = BloodPressureRetryGenerator()

    clinical_case = generator.generate("septic shock")

    assert generator.calls == 2
    assert generator.fact_calls == 1
    assert clinical_case.vitals.blood_pressure == "120/80"
