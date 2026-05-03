from __future__ import annotations

from agents.validator import (
    CORRECTION_SYSTEM_PROMPT,
    MAX_VALIDATOR_ATTEMPTS,
    ValidatorAgent,
    allowed_correction_fields,
)
from schemas.clinical_case import ClinicalCase
from tests.test_clinical_case_schema import sample_case


class StubCorrectionAgent(ValidatorAgent):
    def __init__(self, proposed_case: ClinicalCase) -> None:
        self.proposed_case = proposed_case
        self.last_system_prompt: str | None = None
        self.last_user_prompt: str | None = None

    def parse_response(self, **kwargs: object) -> ClinicalCase:
        self.last_system_prompt = kwargs["system_prompt"]
        self.last_user_prompt = kwargs["user_prompt"]
        return self.proposed_case


class RetryingCorrectionAgent(ValidatorAgent):
    def __init__(self) -> None:
        self.calls = 0

    def parse_response(self, **_: object) -> ClinicalCase:
        self.calls += 1
        payload = sample_case()
        payload["vitals"] = {
            "heart_rate": 118,
            "blood_pressure": "one twenty over eighty" if self.calls == 1 else "120-80",
            "respiratory_rate": 24,
            "oxygen_saturation": 94,
        }
        return ClinicalCase.model_validate(payload)


class AlwaysInvalidCorrectionAgent(ValidatorAgent):
    def __init__(self) -> None:
        self.calls = 0

    def parse_response(self, **_: object) -> ClinicalCase:
        self.calls += 1
        payload = sample_case()
        payload["vitals"]["blood_pressure"] = "one twenty over eighty"
        return ClinicalCase.model_validate(payload)


def build_case() -> ClinicalCase:
    return ClinicalCase.model_validate(sample_case())


def test_correction_prompt_limits_rewrites() -> None:
    assert "only modify fields related to the validation failure" in CORRECTION_SYSTEM_PROMPT
    assert "do not rewrite the entire case" in CORRECTION_SYSTEM_PROMPT
    assert "preserve original structure and intent" in CORRECTION_SYSTEM_PROMPT


def test_allowed_correction_fields_maps_vital_failures_to_vitals_only() -> None:
    fields = allowed_correction_fields(
        "heart_rate is outside reasonable clinical range; shock case has low blood pressure"
    )

    assert fields == {"vitals"}


def test_correct_merges_only_allowed_fields_from_proposed_case() -> None:
    original_case = build_case()
    proposed_payload = sample_case()
    proposed_payload["symptoms"] = ["cough", "fever", "sputum"]
    proposed_payload["vitals"] = {
        "heart_rate": 118,
        "blood_pressure": "96/58",
        "respiratory_rate": 24,
        "oxygen_saturation": 94,
    }
    proposed_case = ClinicalCase.model_validate(proposed_payload)
    agent = StubCorrectionAgent(proposed_case)

    corrected_case = agent.correct(
        original_case,
        failure_reason="heart_rate is outside reasonable clinical range",
    )

    assert corrected_case.vitals.heart_rate == 118
    assert corrected_case.symptoms == original_case.symptoms
    assert agent.last_user_prompt is not None
    assert "Only modify these top-level fields if needed: vitals" in agent.last_user_prompt


def test_correct_retries_after_unrecoverable_blood_pressure_validation_failure() -> None:
    original_case = build_case()
    agent = RetryingCorrectionAgent()

    corrected_case = agent.correct(
        original_case,
        failure_reason="blood_pressure must use systolic/diastolic format",
    )

    assert agent.calls == 2
    assert corrected_case.vitals.blood_pressure == "120/80"


def test_correct_fails_cleanly_after_repeated_invalid_blood_pressure() -> None:
    agent = AlwaysInvalidCorrectionAgent()

    try:
        agent.correct(
            build_case(),
            failure_reason="blood_pressure must use systolic/diastolic format",
        )
    except RuntimeError as exc:
        assert "case validation fallback failed after" in str(exc)
    else:
        raise AssertionError("expected correction to fail after retry attempts")

    assert agent.calls == MAX_VALIDATOR_ATTEMPTS
